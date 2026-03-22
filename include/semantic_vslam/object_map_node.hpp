#pragma once
/**
 * object_map_node.hpp
 *
 * 物体级语义地图节点
 *
 * 从语义点云中提取静态物体的 3D 包围盒 (AABB),
 * 跨帧合并构建物体级地图, 发布 MarkerArray 供 RViz 可视化。
 *
 * 输入:
 *   - /semantic_vslam/semantic_cloud  (PointCloud2)
 *   - /semantic_vslam/label_map       (Image, CV_8UC1)
 *
 * 输出:
 *   - /semantic_vslam/object_markers  (MarkerArray)
 */

#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Core>

namespace semantic_vslam {

// 单个物体实例
struct ObjectInstance {
  int class_id;                  // COCO 类别 ID
  Eigen::Vector3f center;        // 中心坐标 (map 坐标系)
  Eigen::Vector3f size;          // 长宽高 (m)
  int observe_count;             // 累积观测次数
  rclcpp::Time last_seen;        // 最后观测时间
};

class ObjectMapNode : public rclcpp::Node {
public:
  ObjectMapNode();

private:
  // 同步回调: semantic_cloud + label_map 到达时处理
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
  void labelCallback(const sensor_msgs::msg::Image::SharedPtr label_msg);

  // 定时处理 + 发布
  void processTimer();

  // 从点云 + 标签图中提取物体包围盒
  void extractObjects();

  // 发布 MarkerArray
  void publishMarkers();

  // ---- ROS 接口 ----
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr label_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---- TF ----
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ---- 数据缓存 ----
  sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_;
  sensor_msgs::msg::Image::SharedPtr latest_label_;
  std::mutex data_mutex_;

  // ---- 物体地图 ----
  std::vector<ObjectInstance> object_map_;
  std::mutex map_mutex_;

  // ---- 参数 ----
  std::string target_frame_;     // "map"
  int min_points_;               // 物体最少 3D 点数
  float merge_distance_;         // 同类物体合并距离 (m)
  int max_objects_;              // 最大物体数量
  bool enable_profiling_ = false;
};

} // namespace semantic_vslam
