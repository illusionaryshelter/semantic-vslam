#pragma once
/**
 * semantic_map_node.hpp
 *
 * 语义地图累积节点
 *
 * 功能: 订阅每帧语义点云，利用 rtabmap 的 TF 树累积到 map 坐标系。
 *
 * 输出:
 *   - /semantic_vslam/semantic_map_cloud  — 3D 语义着色点云地图
 *   - /semantic_vslam/grid_map            — 2D 栅格 (OccupancyGrid 或彩色 Image)
 */

#include <deque>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

class SemanticMapNode : public rclcpp::Node {
public:
  SemanticMapNode();

private:
  /// 每帧语义点云回调
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  /// 定时发布累积语义地图 (3D + 2D)
  void publishTimer();

  // ---- ROS 接口 ----
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr grid_img_pub_;  // 彩色语义图
  rclcpp::TimerBase::SharedPtr timer_;

  // ---- TF ----
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ---- 累积点云 (sliding window) ----
  struct StampedCloud {
    rclcpp::Time stamp;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  };
  std::deque<StampedCloud> cloud_window_;
  std::mutex mutex_;

  // ---- 参数 ----
  std::string target_frame_;   // "map"
  double voxel_size_;          // 体素滤波尺寸 (m)
  int max_clouds_;             // 滑动窗口帧数
  int cloud_decimation_;       // 输入点云抽稀 (性能优化)
  double grid_cell_size_;      // 2D 栅格分辨率 (m)
  double grid_min_height_;     // 障碍物最低高度 (m)
  double grid_max_height_;     // 障碍物最高高度 (m)
  std::string grid_mode_;      // "occupancy" or "image"
};

} // namespace semantic_vslam
