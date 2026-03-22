/**
 * object_map_node.cpp
 *
 * 物体级语义地图: 从 YOLO 分割 + 深度 → 3D 包围盒 → 跨帧物体地图
 *
 * 核心流程:
 *   1. 接收 semantic_cloud + label_map
 *   2. 按 label 分组提取每个静态物体的 3D 点
 *   3. getMinMax3D 拟合 AABB
 *   4. TF 变换到 map 坐标系
 *   5. 跨帧合并 (同类 + 距离 < merge_distance)
 *   6. 发布 MarkerArray (半透明立方体 + 文字标签)
 */

#include "semantic_vslam/object_map_node.hpp"
#include "semantic_vslam/semantic_colors.hpp"

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <pcl/common/common.h>   // getMinMax3D
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <set>
#include <unordered_map>

namespace semantic_vslam {

// 静态物体类别 — 值得建立 3D 包围盒的 COCO 类
// 大型家具/固定物
static const std::set<int> kStaticObjectClasses = {
    13,  // bench
    56,  // chair
    57,  // couch
    58,  // potted plant
    59,  // bed
    60,  // dining table
    61,  // toilet
    62,  // tv
    72,  // refrigerator
};

// COCO 类名 (用于 Marker 文字标签)
static const char* kCocoNames[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair dryer", "toothbrush"
};

// ---------------------------------------------------------------------------
ObjectMapNode::ObjectMapNode()
    : Node("object_map_node") {

  // ---- 参数 ----
  this->declare_parameter<std::string>("target_frame", "map");
  this->declare_parameter<int>("min_points", 50);
  this->declare_parameter<double>("merge_distance", 0.5);
  this->declare_parameter<int>("max_objects", 50);
  this->declare_parameter<double>("publish_rate", 1.0);

  target_frame_ = this->get_parameter("target_frame").as_string();
  min_points_ = this->get_parameter("min_points").as_int();
  merge_distance_ = static_cast<float>(
      this->get_parameter("merge_distance").as_double());
  max_objects_ = this->get_parameter("max_objects").as_int();
  double publish_rate = this->get_parameter("publish_rate").as_double();

  // ---- TF2 ----
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ---- 订阅 ----
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_cloud", 5,
      std::bind(&ObjectMapNode::cloudCallback, this, std::placeholders::_1));
  label_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/semantic_vslam/label_map", 5,
      std::bind(&ObjectMapNode::labelCallback, this, std::placeholders::_1));

  // ---- 发布 ----
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/semantic_vslam/object_markers", 1);

  // ---- 定时处理 ----
  auto period_ms = static_cast<int>(1000.0 / publish_rate);
  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(period_ms),
      std::bind(&ObjectMapNode::processTimer, this));

  RCLCPP_INFO(this->get_logger(),
      "ObjectMapNode ready. min_pts=%d, merge_dist=%.2f, max_objs=%d",
      min_points_, merge_distance_, max_objects_);
}

// ---------------------------------------------------------------------------
void ObjectMapNode::cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_cloud_ = msg;
}

void ObjectMapNode::labelCallback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_label_ = msg;
}

// ---------------------------------------------------------------------------
void ObjectMapNode::processTimer() {
  extractObjects();
  publishMarkers();
}

// ---------------------------------------------------------------------------
void ObjectMapNode::extractObjects() {
  // 取出最新数据
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
  sensor_msgs::msg::Image::SharedPtr label_msg;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    cloud_msg = latest_cloud_;
    label_msg = latest_label_;
    latest_cloud_.reset();
    latest_label_.reset();
  }

  if (!cloud_msg || !label_msg) return;

  // 时间戳差距过大则跳过 (不是同一帧)
  double dt = std::abs(
      rclcpp::Time(cloud_msg->header.stamp).seconds() -
      rclcpp::Time(label_msg->header.stamp).seconds());
  if (dt > 0.1) return;

  // 查找 TF: cloud frame → map
  Eigen::Matrix4f tf_mat;
  try {
    auto tf_stamped = tf_buffer_->lookupTransform(
        target_frame_, cloud_msg->header.frame_id,
        tf2::TimePointZero, tf2::durationFromSec(0.1));
    Eigen::Isometry3d tf_eigen = tf2::transformToEigen(tf_stamped.transform);
    tf_mat = tf_eigen.matrix().cast<float>();
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
        "TF lookup failed: %s", ex.what());
    return;
  }

  // 解析点云 (organized: width x height)
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::fromROSMsg(*cloud_msg, cloud);

  // 解析 label_map
  cv_bridge::CvImageConstPtr cv_label;
  try {
    cv_label = cv_bridge::toCvShare(label_msg, "mono8");
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge label error: %s", e.what());
    return;
  }
  const cv::Mat &label_map = cv_label->image;

  if (static_cast<int>(cloud.width) != label_map.cols ||
      static_cast<int>(cloud.height) != label_map.rows) {
    return;  // 尺寸不匹配
  }

  const int rows = label_map.rows;
  const int cols = label_map.cols;

  // 按 class_id 分组收集 3D 点 (只处理静态物体类别)
  std::unordered_map<int, std::vector<Eigen::Vector3f>> class_points;

  for (int v = 0; v < rows; ++v) {
    const uint8_t *lbl_row = label_map.ptr<uint8_t>(v);
    const pcl::PointXYZRGB *cloud_row = &cloud.points[v * cols];
    for (int u = 0; u < cols; ++u) {
      uint8_t lbl = lbl_row[u];
      if (lbl == 0) continue;  // 无标签
      int cls = lbl - 1;
      if (!kStaticObjectClasses.count(cls)) continue;  // 非静态物体

      const auto &pt = cloud_row[u];
      if (std::isnan(pt.z) || pt.z <= 0.01f) continue;

      // 变换到 map 坐标系
      Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
      Eigen::Vector4f p_map = tf_mat * p;
      class_points[cls].emplace_back(p_map[0], p_map[1], p_map[2]);
    }
  }

  // 对每个类别拟合 AABB
  rclcpp::Time now_time = this->now();
  std::lock_guard<std::mutex> lock(map_mutex_);

  for (auto &[cls, pts] : class_points) {
    if (static_cast<int>(pts.size()) < min_points_) continue;

    // 计算 AABB
    Eigen::Vector3f min_pt(1e9f, 1e9f, 1e9f);
    Eigen::Vector3f max_pt(-1e9f, -1e9f, -1e9f);
    for (const auto &p : pts) {
      min_pt = min_pt.cwiseMin(p);
      max_pt = max_pt.cwiseMax(p);
    }

    Eigen::Vector3f center = (min_pt + max_pt) * 0.5f;
    Eigen::Vector3f size = max_pt - min_pt;

    // 尺寸合理性检查: 忽略太小 (<5cm) 或太大 (>5m) 的
    if (size.maxCoeff() < 0.05f || size.maxCoeff() > 5.0f) continue;

    // 数据关联: 在已有物体列表中查找同类 + 距离近的
    bool merged = false;
    for (auto &obj : object_map_) {
      if (obj.class_id != cls) continue;
      float dist = (obj.center - center).norm();
      if (dist < merge_distance_) {
        // 加权合并: 已有观测权重更大, 逐渐稳定
        float w = 1.0f / (obj.observe_count + 1);
        obj.center = obj.center * (1.0f - w) + center * w;
        obj.size = obj.size * (1.0f - w) + size * w;
        obj.observe_count++;
        obj.last_seen = now_time;
        merged = true;
        break;
      }
    }

    if (!merged && static_cast<int>(object_map_.size()) < max_objects_) {
      object_map_.push_back({cls, center, size, 1, now_time});
    }
  }

  // 清理长时间未观测的物体 (>30秒)
  object_map_.erase(
      std::remove_if(object_map_.begin(), object_map_.end(),
          [&](const ObjectInstance &obj) {
            return (now_time - obj.last_seen).seconds() > 30.0;
          }),
      object_map_.end());
}

// ---------------------------------------------------------------------------
void ObjectMapNode::publishMarkers() {
  std::lock_guard<std::mutex> lock(map_mutex_);

  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = this->now();

  // 先清除旧 markers
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  delete_marker.header.stamp = stamp;
  delete_marker.header.frame_id = target_frame_;
  marker_array.markers.push_back(delete_marker);

  int id = 0;
  for (const auto &obj : object_map_) {
    // 半透明立方体
    visualization_msgs::msg::Marker cube;
    cube.header.stamp = stamp;
    cube.header.frame_id = target_frame_;
    cube.ns = "object_boxes";
    cube.id = id;
    cube.type = visualization_msgs::msg::Marker::CUBE;
    cube.action = visualization_msgs::msg::Marker::ADD;
    cube.pose.position.x = obj.center.x();
    cube.pose.position.y = obj.center.y();
    cube.pose.position.z = obj.center.z();
    cube.pose.orientation.w = 1.0;
    cube.scale.x = std::max(obj.size.x(), 0.05f);
    cube.scale.y = std::max(obj.size.y(), 0.05f);
    cube.scale.z = std::max(obj.size.z(), 0.05f);

    // 颜色: 使用语义颜色表, 半透明
    if (obj.class_id >= 0 && obj.class_id < 80) {
      cube.color.r = kSemanticColors[obj.class_id][0] / 255.0f;
      cube.color.g = kSemanticColors[obj.class_id][1] / 255.0f;
      cube.color.b = kSemanticColors[obj.class_id][2] / 255.0f;
    } else {
      cube.color.r = 1.0f; cube.color.g = 1.0f; cube.color.b = 0.0f;
    }
    // 观测次数越多越不透明 (1次=0.3, 10次+=0.7)
    cube.color.a = std::min(0.3f + obj.observe_count * 0.04f, 0.7f);
    cube.lifetime = rclcpp::Duration(0, 0);  // 持久
    marker_array.markers.push_back(cube);

    // 文字标签
    visualization_msgs::msg::Marker text;
    text.header = cube.header;
    text.ns = "object_labels";
    text.id = id;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = obj.center.x();
    text.pose.position.y = obj.center.y();
    text.pose.position.z = obj.center.z() + obj.size.z() * 0.5f + 0.1f;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.12;  // 文字高度
    text.color.r = 1.0f; text.color.g = 1.0f;
    text.color.b = 1.0f; text.color.a = 1.0f;

    const char *name = (obj.class_id >= 0 && obj.class_id < 80)
                           ? kCocoNames[obj.class_id] : "unknown";
    text.text = std::string(name) + " (x" +
                std::to_string(obj.observe_count) + ")";
    text.lifetime = rclcpp::Duration(0, 0);
    marker_array.markers.push_back(text);

    id++;
  }

  marker_pub_->publish(marker_array);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "Object map: %zu objects tracked", object_map_.size());
}

} // namespace semantic_vslam

// ---- main ----
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<semantic_vslam::ObjectMapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
