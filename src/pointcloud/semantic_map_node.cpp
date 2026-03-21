/**
 * semantic_map_node.cpp
 *
 * 语义地图累积节点
 *
 * 输出:
 *   1. /semantic_vslam/semantic_map_cloud — 3D 语义着色点云地图
 *   2. /semantic_vslam/grid_map           — 2D 占据栅格地图 (从 3D 点云投影)
 */

#include "semantic_vslam/semantic_map_node.hpp"
#include "semantic_vslam/semantic_colors.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <set>
#include <unordered_map>

namespace semantic_vslam {

// 动态类别 (导航时忽略, 代价=0)
static const std::set<int> kDynamicClasses = {
    0, 1, 2, 3, 5, 6, 7, 14, 15, 16, 17, 18, 19
};

// RGB → class_id 反向查表 (从 kSemanticColors 构建)
static std::unordered_map<uint32_t, int> buildColorToClassMap() {
  std::unordered_map<uint32_t, int> m;
  for (int i = 0; i < 80; ++i) {
    uint32_t key = (static_cast<uint32_t>(kSemanticColors[i][0]) << 16) |
                   (static_cast<uint32_t>(kSemanticColors[i][1]) << 8) |
                    static_cast<uint32_t>(kSemanticColors[i][2]);
    m[key] = i;
  }
  return m;
}
static const auto kColorToClass = buildColorToClassMap();

// 根据 class_id 返回导航代价 (0=空闲, 1-99=代价, 100=不可通行)
static int8_t getSemanticCost(int class_id) {
  if (kDynamicClasses.count(class_id)) return 0;   // 动态物体: 忽略
  switch (class_id) {
    // 大型家具: 不可通行
    case 57: // couch
    case 59: // bed
    case 72: // refrigerator
    case 62: // tv
      return 100;
    // 中型障碍: 高代价
    case 56: // chair     — 可能移开
    case 60: // dining table
    case 61: // toilet
      return 80;
    // 小型物品: 中等代价
    case 58: // potted plant
    case 75: // vase
    case 73: // book
      return 50;
    // 其他静态物体: 障碍
    default: return 100;
  }
}

SemanticMapNode::SemanticMapNode()
    : Node("semantic_map_node") {

  // ---- 参数 ----
  this->declare_parameter<std::string>("target_frame", "map");
  this->declare_parameter<double>("voxel_size", 0.02);
  this->declare_parameter<int>("max_clouds", 50);
  this->declare_parameter<int>("cloud_decimation", 2);
  this->declare_parameter<double>("publish_rate", 1.0);
  this->declare_parameter<double>("grid_cell_size", 0.05);
  this->declare_parameter<double>("grid_min_height", 0.1);
  this->declare_parameter<double>("grid_max_height", 2.0);

  target_frame_ = this->get_parameter("target_frame").as_string();
  voxel_size_ = this->get_parameter("voxel_size").as_double();
  max_clouds_ = this->get_parameter("max_clouds").as_int();
  cloud_decimation_ = this->get_parameter("cloud_decimation").as_int();
  double publish_rate = this->get_parameter("publish_rate").as_double();
  grid_cell_size_ = this->get_parameter("grid_cell_size").as_double();
  grid_min_height_ = this->get_parameter("grid_min_height").as_double();
  grid_max_height_ = this->get_parameter("grid_max_height").as_double();

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
  auto period_ms = static_cast<int>(1000.0 / publish_rate);
  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(period_ms),
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
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cloud_window_.empty()) return;

    size_t total = 0;
    for (const auto &sc : cloud_window_) total += sc.cloud->size();
    merged->reserve(total);

    for (const auto &sc : cloud_window_) {
      *merged += *(sc.cloud);
    }
  }

  if (merged->empty()) return;

  // 体素滤波
  if (voxel_size_ > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(merged);
    vg.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    vg.filter(*filtered);
    merged = filtered;
  }

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
    // 计算边界
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

    // 初始化: -1 = unknown
    std::vector<int8_t> grid(width * height, -1);

    // 投影: 语义代价感知
    // 策略: 取每个栅格中最大代价 (max-cost-wins)
    for (const auto &pt : merged->points) {
      int gx = static_cast<int>((pt.x - xMin) / grid_cell_size_);
      int gy = static_cast<int>((pt.y - yMin) / grid_cell_size_);
      if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;

      int idx = gy * width + gx;

      if (pt.z >= grid_min_height_ && pt.z <= grid_max_height_) {
        // 查找语义类别
        uint32_t rgb_key = (static_cast<uint32_t>(pt.r) << 16) |
                           (static_cast<uint32_t>(pt.g) << 8) |
                            static_cast<uint32_t>(pt.b);
        auto it = kColorToClass.find(rgb_key);
        int8_t cost;
        if (it != kColorToClass.end()) {
          cost = getSemanticCost(it->second);
        } else {
          // 非语义点 (原始相机颜色): 视为普通障碍
          cost = 100;
        }

        // max-cost-wins: 只升级不降级
        if (cost > grid[idx]) {
          grid[idx] = cost;
        }
      } else if (grid[idx] < 0) {
        // 低于障碍高度 + 未被任何障碍标记 → 空闲
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

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "Semantic map: %zu points, %zu frames | grid published",
      merged->size(), cloud_window_.size());
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
