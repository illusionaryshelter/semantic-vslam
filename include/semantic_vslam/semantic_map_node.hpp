#pragma once
/**
 * semantic_map_node.hpp
 *
 * 语义地图累积节点
 *
 * 输出:
 *   - /semantic_vslam/semantic_map_cloud  — 3D 语义着色点云地图
 *   - /semantic_vslam/grid_map            — 2D OccupancyGrid (Nav2 导航)
 *   - /semantic_vslam/grid_map_visual     — 2D 彩色扁平点云 (RViz 可视化)
 */

#include <deque>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
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
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void publishTimer();

  // ---- ROS 接口 ----
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grid_visual_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---- TF ----
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ---- 累积点云 ----
  struct StampedCloud {
    rclcpp::Time stamp;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  };
  std::deque<StampedCloud> cloud_window_;
  std::mutex mutex_;

  // ---- 参数 ----
  std::string target_frame_;
  double voxel_size_;
  int max_clouds_;
  int cloud_decimation_;
  double grid_cell_size_;
  double grid_min_height_;
  double grid_max_height_;
};

} // namespace semantic_vslam
