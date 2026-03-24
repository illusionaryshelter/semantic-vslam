/**
 * cuda_voxel_grid.hpp
 *
 * GPU 加速的 VoxelGrid 下采样, 替换 pcl::VoxelGrid
 *
 * 架构: .cu (CUDA kernel, 无 PCL) + .cpp (PCL 封装)
 * 颜色策略: first-point (语义颜色不能平均)
 *
 * 内存策略 (Jetson UMA):
 *   - GPUPool 中间缓冲区: cudaMalloc (纯 device, GPU L2 cached)
 *   - I/O 缓冲区: cudaHostAllocMapped (pinned uncached = 真零拷贝)
 *   - 持久分配: 一次分配跨调用复用
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
 * @param input       输入点数组 (mapped 或 device 指针)
 * @param num_points  输入点数
 * @param h_output    输出缓冲区 (mapped 或 device 指针)
 * @param max_output  输出缓冲区最大容量
 * @param voxel_size  体素尺寸 (m)
 * @return 输出点数
 */
int cudaVoxelGridFilterRaw(
    const VoxelPoint* input, int num_points,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float min_x = 0, float max_x = 0,
    float min_y = 0, float max_y = 0,
    float min_z = 0, float max_z = 0);

/**
 * 真零拷贝内存分配 (cudaHostAllocMapped)
 *
 * Jetson UMA: pinned + uncached → CPU 写直达物理内存, GPU DMA 直读
 * 无 cache flush/invalidation 开销 (对比 cudaMallocManaged)
 *
 * @param max_points  最大点数
 * @param host_ptr    [out] CPU 端指针 (用于填充/读取)
 * @param dev_ptr     [out] GPU 端指针 (传给 CUDA kernel)
 * @return true=成功
 */
bool cudaVoxelGridAllocZeroCopy(int max_points,
                                VoxelPoint** host_ptr,
                                VoxelPoint** dev_ptr);
void cudaVoxelGridFreeZeroCopy(VoxelPoint* host_ptr);

// ---- 兼容旧 API (deprecated, 保留用于单元测试) ----
VoxelPoint* cudaVoxelGridAllocManaged(int max_points);
void cudaVoxelGridFreeManaged(VoxelPoint* ptr);

}  // namespace semantic_vslam

// ---- PCL 高层封装 (仅在 C++ 编译器可见, NVCC 不编译) ----
#ifndef __CUDACC__
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

/**
 * CUDA VoxelGrid 下采样 (PCL 接口)
 *
 * 内部使用 cudaHostAllocMapped 真零拷贝 + cudaMalloc 持久 GPU 池
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
 */
class CudaIncrementalVoxelGrid {
public:
  explicit CudaIncrementalVoxelGrid(float voxel_size);

  /// 将新帧融合到全局地图
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
