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

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

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
  this->declare_parameter<double>("publish_rate", 2.0);  // 2 Hz 更快响应
  this->declare_parameter<bool>("enable_profiling", false);

  target_frame_ = this->get_parameter("target_frame").as_string();
  min_points_ = this->get_parameter("min_points").as_int();
  merge_distance_ = static_cast<float>(
      this->get_parameter("merge_distance").as_double());
  max_objects_ = this->get_parameter("max_objects").as_int();
  double publish_rate = this->get_parameter("publish_rate").as_double();
  enable_profiling_ = this->get_parameter("enable_profiling").as_bool();

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
  if (enable_profiling_) {
    auto t0 = std::chrono::steady_clock::now();
    extractObjects();
    auto t1 = std::chrono::steady_clock::now();
    publishMarkers();
    auto t2 = std::chrono::steady_clock::now();
    auto ms_extract = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto ms_publish = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto ms_total   = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
    if (ms_extract > 0 || ms_publish > 0) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "[perf] obj: extract=%ldms publish=%ldms total=%ldms | %zu objects",
          ms_extract, ms_publish, ms_total, object_map_.size());
    }
  } else {
    extractObjects();
    publishMarkers();
  }
}

// ---------------------------------------------------------------------------
void ObjectMapNode::extractObjects() {
  // 取出最新数据 (取出后清空, 避免重复处理)
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

  // 查找 TF: cloud frame → map (缩短等待时间以加快响应)
  Eigen::Matrix4f tf_mat;
  try {
    auto tf_stamped = tf_buffer_->lookupTransform(
        target_frame_, cloud_msg->header.frame_id,
        tf2::TimePointZero, tf2::durationFromSec(0.05));
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

  // ---- 3D 欧氏聚类实例分离 ----
  // 比 2D connectedComponents 更准确:
  //   两个物体在图像上相邻但 3D 空间分开 → 正确分离
  //
  // 步骤:
  //   1. 按类别收集所有 3D 点 (map 坐标系)
  //   2. 对每个类别做 PCL EuclideanClusterExtraction
  //   3. 每个聚类 = 一个物体实例

  // 按类别收集 3D 点 (已变换到 map 坐标系)
  struct TaggedPoint {
    Eigen::Vector3f pos;
    int class_id;
  };
  std::unordered_map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> class_clouds;

  for (int v = 0; v < rows; ++v) {
    const uint8_t *lbl_row = label_map.ptr<uint8_t>(v);
    const pcl::PointXYZRGB *cloud_row = &cloud.points[v * cols];
    for (int u = 0; u < cols; ++u) {
      uint8_t lbl = lbl_row[u];
      if (lbl == 0) continue;
      int cls = lbl - 1;
      if (!kStaticObjectClasses.count(cls)) continue;

      const auto &pt = cloud_row[u];
      if (std::isnan(pt.z) || pt.z <= 0.01f) continue;

      Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
      Eigen::Vector4f p_map = tf_mat * p;

      if (!class_clouds.count(cls)) {
        class_clouds[cls] = pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>);
      }
      pcl::PointXYZ pp;
      pp.x = p_map[0]; pp.y = p_map[1]; pp.z = p_map[2];
      class_clouds[cls]->push_back(pp);
    }
  }

  // 对每个类别做 3D 欧氏聚类 → 拟合 AABB
  struct DetectedObject {
    int class_id;
    Eigen::Vector3f center;
    Eigen::Vector3f size;
  };
  std::vector<DetectedObject> detections;

  for (auto &[cls, cls_cloud] : class_clouds) {
    if (static_cast<int>(cls_cloud->size()) < min_points_) continue;

    // KdTree + 欧氏聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cls_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.15);   // 15cm: 同一物体内点间距
    ec.setMinClusterSize(min_points_);
    ec.setMaxClusterSize(50000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cls_cloud);
    ec.extract(cluster_indices);

    for (const auto &indices : cluster_indices) {
      Eigen::Vector3f min_pt(1e9f, 1e9f, 1e9f);
      Eigen::Vector3f max_pt(-1e9f, -1e9f, -1e9f);
      for (int idx : indices.indices) {
        const auto &p = cls_cloud->points[idx];
        Eigen::Vector3f ep(p.x, p.y, p.z);
        min_pt = min_pt.cwiseMin(ep);
        max_pt = max_pt.cwiseMax(ep);
      }
      Eigen::Vector3f center = (min_pt + max_pt) * 0.5f;
      Eigen::Vector3f size = max_pt - min_pt;

      // 尺寸合理性检查
      if (size.maxCoeff() < 0.05f || size.maxCoeff() > 3.0f) continue;
      detections.push_back({cls, center, size});
    }
  }

  // ---- 3D AABB IoU 数据关联 ----
  auto computeIoU = [](const Eigen::Vector3f &c1, const Eigen::Vector3f &s1,
                       const Eigen::Vector3f &c2, const Eigen::Vector3f &s2) -> float {
    Eigen::Vector3f min1 = c1 - s1 * 0.5f, max1 = c1 + s1 * 0.5f;
    Eigen::Vector3f min2 = c2 - s2 * 0.5f, max2 = c2 + s2 * 0.5f;
    Eigen::Vector3f inter_min = min1.cwiseMax(min2);
    Eigen::Vector3f inter_max = max1.cwiseMin(max2);
    Eigen::Vector3f inter_size = (inter_max - inter_min).cwiseMax(0.0f);
    float inter_vol = inter_size.x() * inter_size.y() * inter_size.z();
    float vol1 = s1.x() * s1.y() * s1.z();
    float vol2 = s2.x() * s2.y() * s2.z();
    float union_vol = vol1 + vol2 - inter_vol;
    return (union_vol > 1e-6f) ? (inter_vol / union_vol) : 0.0f;
  };

  auto isContainedIn = [](const Eigen::Vector3f &c1, const Eigen::Vector3f &s1,
                          const Eigen::Vector3f &c2, const Eigen::Vector3f &s2) -> bool {
    Eigen::Vector3f min1 = c1 - s1 * 0.5f, max1 = c1 + s1 * 0.5f;
    Eigen::Vector3f min2 = c2 - s2 * 0.5f, max2 = c2 + s2 * 0.5f;
    Eigen::Vector3f inter_min = min1.cwiseMax(min2);
    Eigen::Vector3f inter_max = max1.cwiseMin(max2);
    Eigen::Vector3f inter_size = (inter_max - inter_min).cwiseMax(0.0f);
    float inter_vol = inter_size.x() * inter_size.y() * inter_size.z();
    float vol1 = s1.x() * s1.y() * s1.z();
    return (vol1 > 1e-6f) && (inter_vol / vol1 > 0.5f);
  };

  rclcpp::Time now_time = this->now();
  std::lock_guard<std::mutex> lock(map_mutex_);

  for (auto &det : detections) {
    // 数据关联: IoU > 0.15 或被包含 → 合并
    int best_idx = -1;
    float best_iou = 0.0f;
    for (int i = 0; i < static_cast<int>(object_map_.size()); ++i) {
      auto &obj = object_map_[i];
      if (obj.class_id != det.class_id) continue;
      float iou = computeIoU(obj.center, obj.size, det.center, det.size);
      bool contained = isContainedIn(det.center, det.size, obj.center, obj.size) ||
                       isContainedIn(obj.center, obj.size, det.center, det.size);
      float score = contained ? std::max(iou, 0.2f) : iou;
      if (score > best_iou) {
        best_iou = score;
        best_idx = i;
      }
    }

    if (best_idx >= 0 && best_iou > 0.15f) {
      auto &obj = object_map_[best_idx];
      float w = 1.0f / (obj.observe_count + 1);
      obj.center = obj.center * (1.0f - w) + det.center * w;
      obj.size = obj.size * (1.0f - w) + det.size * w;
      obj.observe_count++;
      obj.last_seen = now_time;
    } else if (static_cast<int>(object_map_.size()) < max_objects_) {
      object_map_.push_back({det.class_id, det.center, det.size, 1, now_time});
    }
  }

  // ---- 合并冗余物体 (同类 + 重叠显著) ----
  for (int i = 0; i < static_cast<int>(object_map_.size()); ++i) {
    for (int j = i + 1; j < static_cast<int>(object_map_.size()); ) {
      if (object_map_[i].class_id != object_map_[j].class_id) { ++j; continue; }
      float iou = computeIoU(object_map_[i].center, object_map_[i].size,
                             object_map_[j].center, object_map_[j].size);
      bool contained = isContainedIn(object_map_[j].center, object_map_[j].size,
                                     object_map_[i].center, object_map_[i].size) ||
                       isContainedIn(object_map_[i].center, object_map_[i].size,
                                     object_map_[j].center, object_map_[j].size);
      if (iou > 0.1f || contained) {
        auto &keep = (object_map_[i].observe_count >= object_map_[j].observe_count)
                         ? object_map_[i] : object_map_[j];
        auto &drop = (object_map_[i].observe_count >= object_map_[j].observe_count)
                         ? object_map_[j] : object_map_[i];
        float w = static_cast<float>(drop.observe_count) /
                  (keep.observe_count + drop.observe_count);
        keep.center = keep.center * (1.0f - w) + drop.center * w;
        keep.size = keep.size.cwiseMax(drop.size);
        keep.observe_count += drop.observe_count;
        if (drop.last_seen > keep.last_seen) keep.last_seen = drop.last_seen;
        object_map_.erase(object_map_.begin() + j);
      } else {
        ++j;
      }
    }
  }

  // ---- 观测计数衰减 + 清理 ----
  // 超过 5 秒未观测 → observe_count 每周期 -1 (渐进淘汰)
  for (auto &obj : object_map_) {
    if ((now_time - obj.last_seen).seconds() > 5.0 && obj.observe_count > 0) {
      obj.observe_count--;
    }
  }
  // 超过 15 秒未观测 或 observe_count 降为 0 → 移除
  object_map_.erase(
      std::remove_if(object_map_.begin(), object_map_.end(),
          [&](const ObjectInstance &obj) {
            return (now_time - obj.last_seen).seconds() > 15.0 ||
                   obj.observe_count <= 0;
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
    // 至少观测 2 次才显示 (过滤噪声)
    if (obj.observe_count < 1) continue;  // 首次检测即显示

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
