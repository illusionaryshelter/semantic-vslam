/**
 * semantic_map_node.cpp
 *
 * 语义地图累积节点 — 体素哈希表累积器
 *
 * 核心思路:
 *   1. 每帧回调: 将新点插入/更新体素哈希表 (O(新帧点数))
 *   2. 定时发布: 序列化哈希表 → PointCloud2 + OccupancyGrid
 *   3. 定时清理: 删除超过 TTL 的旧体素
 *
 * 优势:
 *   - 持久化地图 (不会因转头而消失)
 *   - 零重影 (同一体素位置只有一个点)
 *   - 回调 O(n), 发布 O(地图大小), 无需额外 VoxelGrid 滤波
 */

#include "semantic_vslam/semantic_map_node.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace semantic_vslam {

SemanticMapNode::SemanticMapNode()
    : Node("semantic_map_node") {

  // ---- 参数 ----
  this->declare_parameter<std::string>("target_frame", "map");
  this->declare_parameter<double>("voxel_size", 0.02);
  this->declare_parameter<int>("cloud_decimation", 2);
  this->declare_parameter<double>("publish_rate", 1.0);
  this->declare_parameter<double>("grid_cell_size", 0.05);
  this->declare_parameter<double>("grid_min_height", 0.1);
  this->declare_parameter<double>("grid_max_height", 2.0);
  this->declare_parameter<double>("voxel_ttl", 120.0);  // 体素存活 120 秒
  this->declare_parameter<bool>("enable_profiling", false);

  target_frame_ = this->get_parameter("target_frame").as_string();
  voxel_size_ = this->get_parameter("voxel_size").as_double();
  inv_voxel_size_ = 1.0 / voxel_size_;
  cloud_decimation_ = this->get_parameter("cloud_decimation").as_int();
  double publish_rate = this->get_parameter("publish_rate").as_double();
  grid_cell_size_ = this->get_parameter("grid_cell_size").as_double();
  grid_min_height_ = this->get_parameter("grid_min_height").as_double();
  grid_max_height_ = this->get_parameter("grid_max_height").as_double();
  voxel_ttl_ = this->get_parameter("voxel_ttl").as_double();
  enable_profiling_ = this->get_parameter("enable_profiling").as_bool();

  // ---- TF2 ----
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ---- 订阅语义点云 ----
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_cloud", 5,
      std::bind(&SemanticMapNode::cloudCallback, this, std::placeholders::_1));

  // ---- 发布 3D 语义地图 + 2D 栅格地图 ----
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_map_cloud", 1);
  grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/semantic_vslam/grid_map", 1);

  // ---- 定时发布 ----
  auto period = std::chrono::milliseconds(
      static_cast<int>(1000.0 / publish_rate));
  timer_ = this->create_wall_timer(period,
      std::bind(&SemanticMapNode::publishTimer, this));

  // 预分配哈希表
  voxel_map_.reserve(100000);

  RCLCPP_INFO(this->get_logger(),
      "SemanticMapNode ready. voxel=%.3f, grid_cell=%.3f, ttl=%.0fs",
      voxel_size_, grid_cell_size_, voxel_ttl_);
}

// ---------------------------------------------------------------------------
void SemanticMapNode::cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

  // 查找 TF: semantic_cloud frame → map
  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    tf_stamped = tf_buffer_->lookupTransform(
        target_frame_, msg->header.frame_id,
        tf2::TimePointZero,
        tf2::durationFromSec(0.1));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
        "TF lookup failed: %s", ex.what());
    return;
  }

  Eigen::Isometry3d tf_eigen = tf2::transformToEigen(tf_stamped.transform);
  Eigen::Matrix4f tf_mat = tf_eigen.matrix().cast<float>();

  // 解析输入点云
  pcl::PointCloud<pcl::PointXYZRGB> input_cloud;
  pcl::fromROSMsg(*msg, input_cloud);

  const int dec = std::max(1, cloud_decimation_);
  const int total = static_cast<int>(input_cloud.size());
  const double now_sec = this->now().seconds();
  const float inv_vs = static_cast<float>(inv_voxel_size_);

  // ---- 逐点插入/更新体素哈希表 ----
  std::lock_guard<std::mutex> lock(mutex_);

  for (int i = 0; i < total; i += dec) {
    const auto &pt = input_cloud.points[i];
    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
      continue;
    if (pt.z <= 0.0f) continue;

    // 变换到 map 坐标系
    Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
    Eigen::Vector4f p_map = tf_mat * p;

    // 离散化为体素坐标
    VoxelKey key;
    key.x = static_cast<int>(std::floor(p_map[0] * inv_vs));
    key.y = static_cast<int>(std::floor(p_map[1] * inv_vs));
    key.z = static_cast<int>(std::floor(p_map[2] * inv_vs));

    // 插入或更新
    auto it = voxel_map_.find(key);
    if (it != voxel_map_.end()) {
      // 增量加权平均: 新观测权重 = 1/(count+1)
      auto &v = it->second;
      float w = 1.0f / (v.count + 1);
      v.x = v.x * (1.0f - w) + p_map[0] * w;
      v.y = v.y * (1.0f - w) + p_map[1] * w;
      v.z = v.z * (1.0f - w) + p_map[2] * w;
      v.r = v.r * (1.0f - w) + static_cast<float>(pt.r) * w;
      v.g = v.g * (1.0f - w) + static_cast<float>(pt.g) * w;
      v.b = v.b * (1.0f - w) + static_cast<float>(pt.b) * w;
      v.count++;
      v.last_update = now_sec;
    } else {
      voxel_map_[key] = {
        static_cast<float>(pt.r), static_cast<float>(pt.g),
        static_cast<float>(pt.b),
        p_map[0], p_map[1], p_map[2],
        1, now_sec
      };
    }
  }
}

// ---------------------------------------------------------------------------
void SemanticMapNode::publishTimer() {
  auto tp0 = std::chrono::steady_clock::now();

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (voxel_map_.empty()) return;

    const double now_sec = this->now().seconds();

    // 清理过期体素 + 序列化存活的
    cloud->reserve(voxel_map_.size());
    for (auto it = voxel_map_.begin(); it != voxel_map_.end(); ) {
      if (now_sec - it->second.last_update > voxel_ttl_) {
        it = voxel_map_.erase(it);
      } else {
        pcl::PointXYZRGB pt;
        pt.x = it->second.x;
        pt.y = it->second.y;
        pt.z = it->second.z;
        pt.r = static_cast<uint8_t>(std::clamp(it->second.r, 0.0f, 255.0f));
        pt.g = static_cast<uint8_t>(std::clamp(it->second.g, 0.0f, 255.0f));
        pt.b = static_cast<uint8_t>(std::clamp(it->second.b, 0.0f, 255.0f));
        cloud->push_back(pt);
        ++it;
      }
    }
  }

  if (cloud->empty()) return;

  auto tp1 = std::chrono::steady_clock::now();

  // ---- 发布 3D 语义地图 ----
  {
    sensor_msgs::msg::PointCloud2 pc2;
    pcl::toROSMsg(*cloud, pc2);
    pc2.header.stamp = this->now();
    pc2.header.frame_id = target_frame_;
    map_pub_->publish(pc2);
  }

  // ---- 发布 2D 占据栅格地图 (从 3D 点云投影) ----
  {
    float xMin = 1e9f, xMax = -1e9f, yMin = 1e9f, yMax = -1e9f;
    for (const auto &pt : cloud->points) {
      if (pt.x < xMin) xMin = pt.x;
      if (pt.x > xMax) xMax = pt.x;
      if (pt.y < yMin) yMin = pt.y;
      if (pt.y > yMax) yMax = pt.y;
    }
    xMin -= grid_cell_size_;  yMin -= grid_cell_size_;
    xMax += grid_cell_size_;  yMax += grid_cell_size_;

    int width  = static_cast<int>((xMax - xMin) / grid_cell_size_) + 1;
    int height = static_cast<int>((yMax - yMin) / grid_cell_size_) + 1;

    if (width <= 0 || height <= 0 || width > 10000 || height > 10000)
      return;

    std::vector<int8_t> grid(width * height, -1);

    for (const auto &pt : cloud->points) {
      int gx = static_cast<int>((pt.x - xMin) / grid_cell_size_);
      int gy = static_cast<int>((pt.y - yMin) / grid_cell_size_);
      if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;

      int idx = gy * width + gx;
      if (pt.z >= grid_min_height_ && pt.z <= grid_max_height_) {
        grid[idx] = 100;
      } else if (grid[idx] != 100) {
        grid[idx] = 0;
      }
    }

    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header.stamp = this->now();
    grid_msg.header.frame_id = target_frame_;
    grid_msg.info.resolution = grid_cell_size_;
    grid_msg.info.width = width;
    grid_msg.info.height = height;
    grid_msg.info.origin.position.x = xMin;
    grid_msg.info.origin.position.y = yMin;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;
    grid_msg.data.assign(grid.begin(), grid.end());
    grid_pub_->publish(grid_msg);
  }

  if (enable_profiling_) {
    auto tp2 = std::chrono::steady_clock::now();
    auto ms_build = std::chrono::duration_cast<std::chrono::milliseconds>(tp1 - tp0).count();
    auto ms_pub   = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp0).count();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "[perf] map: build=%ldms pub=%ldms total=%ldms | %zu voxels",
        ms_build, ms_pub, ms_total, cloud->size());
  }
}

} // namespace semantic_vslam

// ---- main ----
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<semantic_vslam::SemanticMapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
