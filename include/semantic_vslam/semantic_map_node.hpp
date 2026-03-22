#pragma once
/**
 * semantic_map_node.hpp
 *
 * 语义地图累积节点 — 体素哈希表累积器
 *
 * 使用空间哈希表 (VoxelHashMap) 持久化存储地图:
 *   - 新点插入/更新已有体素 (颜色加权平均)
 *   - 定时清理长时间未更新的体素
 *   - 发布时直接序列化哈希表 → 零额外滤波开销
 *
 * 优势 vs 滑动窗口:
 *   - 地图持久化 (不会因帧推出窗口而消失)
 *   - 零重影 (同一体素只有一个点)
 *   - O(新帧点数) 每次回调, 发布时 O(地图大小)
 */

#include <mutex>
#include <unordered_map>

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
  /// 每帧语义点云回调
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  /// 定时发布累积语义地图 (3D + 2D)
  void publishTimer();

  // ---- ROS 接口 ----
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---- TF ----
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ---- 体素哈希表 ----
  struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey &o) const {
      return x == o.x && y == o.y && z == o.z;
    }
  };
  struct VoxelKeyHash {
    size_t operator()(const VoxelKey &k) const {
      // FNV-1a 风格哈希
      size_t h = 2166136261u;
      h ^= std::hash<int>()(k.x); h *= 16777619u;
      h ^= std::hash<int>()(k.y); h *= 16777619u;
      h ^= std::hash<int>()(k.z); h *= 16777619u;
      return h;
    }
  };
  struct VoxelData {
    float r, g, b;       // 累积颜色 (加权)
    float x, y, z;       // 体素内点的均值坐标
    int count;           // 累积观测计数
    double last_update;  // 最后更新时间 (秒)
  };
  std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> voxel_map_;
  std::mutex mutex_;

  // ---- 参数 ----
  std::string target_frame_;   // "map"
  double voxel_size_;          // 体素尺寸 (m)
  double inv_voxel_size_;      // 1.0 / voxel_size_ (避免重复除法)
  int cloud_decimation_;       // 输入点云抽稀
  double grid_cell_size_;      // 2D 栅格分辨率 (m)
  double grid_min_height_;     // 障碍物最低高度 (m)
  double grid_max_height_;     // 障碍物最高高度 (m)
  double voxel_ttl_;           // 体素存活时间 (秒)
  bool enable_profiling_ = false;
};

} // namespace semantic_vslam
