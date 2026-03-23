#pragma once

/*
 * semantic_cloud_node.hpp
 *
 * ROS2 节点: 订阅 Astra Pro 的 RGB + Depth 图像，利用 YOLOv8-seg
 * 推理生成语义点云 + 物体级 3D 包围盒。
 *
 * 物体检测通过 Masked Back-Projection 实现:
 *   YOLO 实例 mask + depth 反投影 → 3D AABB (无需 PCL 聚类)
 */

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "semantic_vslam/semantic_colors.hpp"
#include "semantic_vslam/yolo_inference.hpp"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace semantic_vslam {

// 物体实例 (跨帧跟踪)
struct ObjectInstance {
  int class_id;                  // COCO 类别 ID
  Eigen::Vector3f center;        // 中心坐标 (map 坐标系)
  Eigen::Vector3f size;          // 长宽高 (m)
  int observe_count;             // 累积观测次数
  rclcpp::Time last_seen;        // 最后观测时间
};

// ---------------------------------------------------------------------------
// SemanticCloudNode
// ---------------------------------------------------------------------------
class SemanticCloudNode : public rclcpp::Node {
public:
  explicit SemanticCloudNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~SemanticCloudNode() override = default;

private:
  // 同步回调
  void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
                    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg);

  // 相机内参回调 (只取一次)
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  // 核心: 将 RGB + Depth + YOLO 掩码 → PointXYZRGB 点云 + 语义标签图
  void generateSemanticCloud(const cv::Mat &rgb, const cv::Mat &depth,
                             const std::vector<Object> &objects,
                             pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                             cv::Mat &out_label_map);

  // Masked Back-Projection: YOLO mask + depth → 3D AABB
  void computeObjectBoxes(const std::vector<Object> &objects,
                          const cv::Mat &depth,
                          const Eigen::Matrix4f &tf_mat);

  // 发布 MarkerArray
  void publishMarkers();

  // YOLO
  std::unique_ptr<YoloInference> yolo_;

  // ROS2 通信
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                      sensor_msgs::msg::Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<Synchronizer> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  // 语义标签图发布 (CV_8UC1, 0=无标签, >0 = class_id+1)
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr label_map_pub_;

  // 用于转发给 RTAB-Map 的原始 image 发布
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;

  // 物体 Marker 发布
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // TF2 (用于 masked back-projection → map 坐标系)
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // 物体跟踪地图
  std::vector<ObjectInstance> object_map_;
  int max_objects_ = 50;
  int min_obj_points_ = 50;  // 物体最少 3D 点数
  std::string target_frame_ = "map";

  // 相机内参
  bool has_cam_info_ = false;
  float fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;

  // 深度尺度 (Astra Pro 深度图单位: mm → m)
  float depth_scale_ = 0.001f;

  // 推理参数
  float conf_thresh_ = 0.4f;
  float iou_thresh_ = 0.45f;
  bool enable_profiling_ = false;
};

} // namespace semantic_vslam
