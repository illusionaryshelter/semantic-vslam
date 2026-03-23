/**
 * cuda_voxel_grid.hpp
 *
 * GPU 加速的 VoxelGrid 下采样, 替换 pcl::VoxelGrid (700ms → ~15ms)
 *
 * 架构: .cu (CUDA kernel, 无 PCL) + .cpp (PCL 封装)
 * 颜色策略: first-point (语义颜色不能平均)
 */

#pragma once

#include <cstdint>

namespace semantic_vslam {

// GPU 端点结构 (紧凑, 无 Eigen/PCL 依赖)
struct VoxelPoint {
  float x, y, z;
  uint8_t r, g, b, pad;
};

/**
 * CUDA 底层实现 (纯 C 接口, 在 .cu 中定义)
 * @return 输出点数
 */
int cudaVoxelGridFilterRaw(
    const VoxelPoint* h_input, int num_points,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z);

}  // namespace semantic_vslam

// ---- PCL 高层封装 (仅在 C++ 编译器可见, NVCC 不编译) ----
#ifndef __CUDACC__
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

/**
 * CUDA VoxelGrid 下采样 (PCL 接口)
 */
void cudaVoxelGridFilter(
    const pcl::PointCloud<pcl::PointXYZRGB>& input,
    pcl::PointCloud<pcl::PointXYZRGB>& output,
    float voxel_size);

}  // namespace semantic_vslam
#endif
