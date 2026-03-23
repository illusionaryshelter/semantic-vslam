/**
 * semantic_map_node.cpp  (ROS 2 Component)
 *
 * 语义地图累积节点
 *
 * 输出:
 *   1. /semantic_vslam/semantic_map_cloud — 3D 语义着色点云地图
 *   2. /semantic_vslam/grid_map           — 2D 占据栅格地图 (从 3D 点云投影)
 */

#include "semantic_vslam/semantic_map_node.hpp"

#include "semantic_vslam/cuda_voxel_grid.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace semantic_vslam {

SemanticMapNode::SemanticMapNode(const rclcpp::NodeOptions &options)
    : Node("semantic_map_node", options) {

  // ---- 参数 ----
  this->declare_parameter<std::string>("target_frame", "map");
  this->declare_parameter<double>("voxel_size", 0.02);
  this->declare_parameter<int>("max_clouds", 50);  // 保留兼容性但不再使用
  this->declare_parameter<int>("cloud_decimation", 2);
  this->declare_parameter<double>("publish_rate", 1.0);
  this->declare_parameter<double>("grid_cell_size", 0.05);
  this->declare_parameter<double>("grid_min_height", 0.1);
  this->declare_parameter<double>("grid_max_height", 2.0);
  this->declare_parameter<bool>("enable_profiling", false);

  target_frame_ = this->get_parameter("target_frame").as_string();
  voxel_size_ = this->get_parameter("voxel_size").as_double();
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

  // ---- 初始化增量式全局地图 ----
  incremental_grid_ = std::make_unique<CudaIncrementalVoxelGrid>(voxel_size_);

  RCLCPP_INFO(this->get_logger(),
      "SemanticMapNode ready (incremental). voxel=%.3f, grid_cell=%.3f",
      voxel_size_, grid_cell_size_);
}

// ---------------------------------------------------------------------------
void SemanticMapNode::cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

  auto t0 = enable_profiling_ ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

  // 查找 TF: semantic_cloud frame → map
  // 关键: 必须用消息时间戳查 TF, 不能用 TimePointZero (最新 TF)
  //       否则快速旋转时位姿偏移 → 重影
  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    // 优先使用精确时间戳 (等待 50ms 让 TF 可用)
    tf_stamped = tf_buffer_->lookupTransform(
        target_frame_, msg->header.frame_id,
        msg->header.stamp,
        tf2::durationFromSec(0.05));
  } catch (const tf2::TransformException &ex) {
    // 精确时间戳不可用时, 跳过此帧 (不用 TimePointZero 替代!)
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
        "TF lookup failed (ts=%u.%u): %s",
        msg->header.stamp.sec, msg->header.stamp.nanosec, ex.what());
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

  // 存储待融合帧 (线程安全: callback → timer)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_cloud_) {
      pending_cloud_ = transformed;
    } else {
      // 如果 timer 还没来得及处理, 合并多帧
      *pending_cloud_ += *transformed;
    }
  }
}

// ---------------------------------------------------------------------------
void SemanticMapNode::publishTimer() {
  auto tp0 = std::chrono::steady_clock::now();

  // 取出待融合帧
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_cloud;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    new_cloud = pending_cloud_;
    pending_cloud_.reset();
  }

  // 增量融合到全局地图
  if (new_cloud && !new_cloud->empty()) {
    incremental_grid_->addCloud(*new_cloud);
  }

  const auto& map = incremental_grid_->getMap();
  if (map.empty()) return;

  auto tp1 = std::chrono::steady_clock::now();

  // ---- 发布 3D 语义地图 ----
  {
    sensor_msgs::msg::PointCloud2 pc2;
    pcl::toROSMsg(map, pc2);
    pc2.header.stamp = this->now();
    pc2.header.frame_id = target_frame_;
    map_pub_->publish(pc2);
  }

  // ---- 发布 2D 占据栅格地图 (从 3D 点云投影) ----
  {
    float xMin = 1e9f, xMax = -1e9f, yMin = 1e9f, yMax = -1e9f;
    for (const auto &pt : map.points) {
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

    for (const auto &pt : map.points) {
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
    auto ms_add = std::chrono::duration_cast<std::chrono::milliseconds>(tp1 - tp0).count();
    auto ms_pub = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp0).count();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "[perf] map: add=%ldms pub=%ldms total=%ldms | %zu pts (global)",
        ms_add, ms_pub, ms_total, map.size());
  }
}

} // namespace semantic_vslam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_vslam::SemanticMapNode)
