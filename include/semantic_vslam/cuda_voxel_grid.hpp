/**
 * cuda_voxel_grid.hpp
 *
 * GPU 加速的 VoxelGrid 下采样, 替换 pcl::VoxelGrid
 *
 * 架构: .cu (CUDA kernel, 无 PCL) + .cpp (PCL 封装)
 * 颜色策略: first-point (语义颜色不能平均)
 *
 * Jetson 优化:
 *   - cudaMallocManaged: UMA 零拷贝 (CPU/GPU 共享物理内存)
 *   - 持久显存池: 避免反复 cudaMalloc/Free
 */

#pragma once

#include <cstdint>

namespace semantic_vslam {

// GPU 端点结构 (紧凑 16 字节, 无 Eigen/PCL 依赖)
struct VoxelPoint {
  float x, y, z;
  uint8_t r, g, b, pad;
};

/**
 * CUDA 底层实现 (纯 C 接口, 在 .cu 中定义)
 *
 * @param input       输入点数组 (可以是 managed 或普通 host 指针)
 * @param num_points  输入点数
 * @param h_output    输出缓冲区 (可以是 managed 或普通 host 指针)
 * @param max_output  输出缓冲区最大容量
 * @param voxel_size  体素尺寸 (m)
 * @return 输出点数
 *
 * 注: bounding box 参数保留兼容性但不再使用 (uint64_t bit-packing 不需要)
 */
int cudaVoxelGridFilterRaw(
    const VoxelPoint* input, int num_points,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float min_x = 0, float max_x = 0,
    float min_y = 0, float max_y = 0,
    float min_z = 0, float max_z = 0);

/**
 * Managed 内存分配/释放 (Jetson UMA 零拷贝)
 *
 * 分配的内存可同时被 CPU 和 GPU 访问, 无需 H2D/D2H 拷贝
 * 用法:
 *   VoxelPoint* buf = cudaVoxelGridAllocManaged(N);
 *   // CPU 直接写入 buf[i] = {x, y, z, r, g, b, 0};
 *   // GPU kernel 直接读取 buf[i]
 *   cudaVoxelGridFreeManaged(buf);
 */
VoxelPoint* cudaVoxelGridAllocManaged(int max_points);
void cudaVoxelGridFreeManaged(VoxelPoint* ptr);

}  // namespace semantic_vslam

// ---- PCL 高层封装 (仅在 C++ 编译器可见, NVCC 不编译) ----
#ifndef __CUDACC__
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

/**
 * CUDA VoxelGrid 下采样 (PCL 接口, 单帧)
 *
 * 内部使用 managed memory + 持久显存池, Jetson UMA 上零拷贝
 */
void cudaVoxelGridFilter(
    const pcl::PointCloud<pcl::PointXYZRGB>& input,
    pcl::PointCloud<pcl::PointXYZRGB>& output,
    float voxel_size);

/**
 * 增量式 CUDA VoxelGrid (全局地图)
 *
 * 维护持久化全局点云地图, 增量融合新帧:
 *   global_map (50K) + new_frame (5K) → cudaVoxelGrid → 50K
 *
 * 对比滑动窗口:
 *   - 滑动窗口: 150帧 merge(120ms) + voxel(200ms), 地图会消失
 *   - 增量式:   55K pts voxel(~25ms), 地图永久保留
 *
 * 颜色策略: first-point (已有体素的颜色不会被新帧覆盖)
 *
 * 用法:
 *   CudaIncrementalVoxelGrid grid(0.02f);
 *   grid.addCloud(new_frame);           // 每帧调用
 *   const auto& map = grid.getMap();    // 获取全局地图
 *   grid.clear();                       // 重置 (可选)
 */
class CudaIncrementalVoxelGrid {
public:
  explicit CudaIncrementalVoxelGrid(float voxel_size);

  /**
   * 将新帧融合到全局地图
   *
   * 内部流程:
   *   1. 拼接 global_map_ + new_cloud (managed memory, 无 150 帧 merge 开销)
   *   2. CUDA sort_by_key + dedup (uint64_t 空间哈希, 无碰撞)
   *   3. 结果覆盖 global_map_
   *
   * 复杂度: O((M+N) log(M+N)), M=全局地图点数, N=新帧点数
   */
  void addCloud(const pcl::PointCloud<pcl::PointXYZRGB>& new_cloud);

  /// 获取当前全局地图 (只读引用)
  const pcl::PointCloud<pcl::PointXYZRGB>& getMap() const { return global_map_; }

  /// 获取当前地图点数
  size_t size() const { return global_map_.size(); }

  /// 清空全局地图
  void clear();

private:
  float voxel_size_;
  pcl::PointCloud<pcl::PointXYZRGB> global_map_;
};

}  // namespace semantic_vslam
#endif
