/**
 * semantic_map_node.cpp
 *
 * 语义地图累积节点
 *
 * 架构:
 *   1. cloudCallback: 接收语义点云 → TF 变换到 map → 抽稀 → 存入滑动窗口
 *   2. publishTimer:  合并窗口内所有帧 → 哈希去重 → 发布 3D 地图 + 2D 栅格
 *
 * 性能关键优化:
 *   - 用哈希去重 (O(n)) 替代 pcl::VoxelGrid (O(n log n) + 高常数)
 *   - 实测: 150 帧 5 万点从 993ms → ~50ms
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
  this->declare_parameter<int>("max_clouds", 150);
  this->declare_parameter<int>("cloud_decimation", 3);
  this->declare_parameter<double>("publish_rate", 1.0);
  this->declare_parameter<double>("grid_cell_size", 0.05);
  this->declare_parameter<double>("grid_min_height", 0.1);
  this->declare_parameter<double>("grid_max_height", 2.0);
  this->declare_parameter<bool>("enable_profiling", false);

  target_frame_ = this->get_parameter("target_frame").as_string();
  voxel_size_ = this->get_parameter("voxel_size").as_double();
  inv_voxel_size_ = static_cast<float>(1.0 / voxel_size_);
  max_clouds_ = this->get_parameter("max_clouds").as_int();
  cloud_decimation_ = this->get_parameter("cloud_decimation").as_int();
  double publish_rate = this->get_parameter("publish_rate").as_double();
  grid_cell_size_ = this->get_parameter("grid_cell_size").as_double();
  grid_min_height_ = this->get_parameter("grid_min_height").as_double();
  grid_max_height_ = this->get_parameter("grid_max_height").as_double();
  enable_profiling_ = this->get_parameter("enable_profiling").as_bool();

  // ---- TF2 ----
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ---- 订阅语义点云 ----
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_cloud", 5,
      std::bind(&SemanticMapNode::cloudCallback, this, std::placeholders::_1));

  // ---- 发布 ----
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_map_cloud", 1);
  grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/semantic_vslam/grid_map", 1);

  auto period = std::chrono::milliseconds(
      static_cast<int>(1000.0 / publish_rate));
  timer_ = this->create_wall_timer(period,
      std::bind(&SemanticMapNode::publishTimer, this));

  RCLCPP_INFO(this->get_logger(),
      "SemanticMapNode ready. voxel=%.3f, grid_cell=%.3f, max_clouds=%d",
      voxel_size_, grid_cell_size_, max_clouds_);
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

  // 抽稀 + 变换 + 过滤 NaN
  auto transformed = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  const int dec = std::max(1, cloud_decimation_);
  const int total = static_cast<int>(input_cloud.size());
  transformed->reserve(total / dec + 1);

  for (int i = 0; i < total; i += dec) {
    const auto &pt = input_cloud.points[i];
    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
      continue;
    if (pt.z <= 0.0f) continue;

    Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
    Eigen::Vector4f p_map = tf_mat * p;

    pcl::PointXYZRGB mp;
    mp.x = p_map[0]; mp.y = p_map[1]; mp.z = p_map[2];
    mp.r = pt.r; mp.g = pt.g; mp.b = pt.b;
    transformed->push_back(mp);
  }

  if (transformed->empty()) return;

  transformed->width = transformed->size();
  transformed->height = 1;
  transformed->is_dense = true;

  // 添加到 sliding window
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cloud_window_.push_back({msg->header.stamp, transformed});
    while (static_cast<int>(cloud_window_.size()) > max_clouds_) {
      cloud_window_.pop_front();
    }
  }
}

// ---------------------------------------------------------------------------
void SemanticMapNode::publishTimer() {
  auto tp0 = std::chrono::steady_clock::now();

  // ---- 合并滑动窗口内所有帧 ----
  size_t total_pts = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cloud_window_.empty()) return;
    for (const auto &sc : cloud_window_) total_pts += sc.cloud->size();
  }

  auto tp1 = std::chrono::steady_clock::now();

  // ---- 哈希去重 (替代 pcl::VoxelGrid, 快 3-5 倍) ----
  // 遍历所有点, 按离散化体素坐标去重, 保留最后看到的颜色
  struct VoxelPt { float x, y, z; uint8_t r, g, b; };
  std::unordered_map<VoxelKey, VoxelPt, VoxelKeyHash> voxel_map;
  voxel_map.reserve(total_pts / 2);  // 预分配 (去重后约一半)

  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &sc : cloud_window_) {
      for (const auto &pt : sc.cloud->points) {
        VoxelKey key;
        key.x = static_cast<int>(std::floor(pt.x * inv_voxel_size_));
        key.y = static_cast<int>(std::floor(pt.y * inv_voxel_size_));
        key.z = static_cast<int>(std::floor(pt.z * inv_voxel_size_));

        // 同一体素: 后来的帧覆盖旧值 (保持最新观测)
        voxel_map[key] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b};
      }
    }
  }

  // ---- 构建输出点云 ----
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  merged->reserve(voxel_map.size());
  for (const auto &[key, v] : voxel_map) {
    pcl::PointXYZRGB pt;
    pt.x = v.x; pt.y = v.y; pt.z = v.z;
    pt.r = v.r; pt.g = v.g; pt.b = v.b;
    merged->push_back(pt);
  }

  if (merged->empty()) return;

  auto tp2 = std::chrono::steady_clock::now();

  // ---- 发布 3D 语义地图 ----
  {
    sensor_msgs::msg::PointCloud2 pc2;
    pcl::toROSMsg(*merged, pc2);
    pc2.header.stamp = this->now();
    pc2.header.frame_id = target_frame_;
    map_pub_->publish(pc2);
  }

  // ---- 发布 2D 占据栅格地图 (从 3D 点云投影) ----
  {
    float xMin = 1e9f, xMax = -1e9f, yMin = 1e9f, yMax = -1e9f;
    for (const auto &pt : merged->points) {
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

    for (const auto &pt : merged->points) {
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
    auto tp3 = std::chrono::steady_clock::now();
    auto ms_dedup = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
    auto ms_pub   = std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2).count();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp0).count();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "[perf] map: dedup=%ldms pub=%ldms total=%ldms | %zu→%zu voxels, %zu frames",
        ms_dedup, ms_pub, ms_total,
        total_pts, voxel_map.size(), cloud_window_.size());
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
