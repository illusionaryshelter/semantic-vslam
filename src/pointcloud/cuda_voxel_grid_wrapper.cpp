/**
 * cuda_voxel_grid_wrapper.cpp
 *
 * PCL ↔ VoxelPoint 转换封装
 * 此文件由 g++ 编译 (可用 PCL/Eigen), 调用 .cu 中的 CUDA 接口
 *
 * 内存策略: cudaHostAllocMapped (Jetson UMA 真零拷贝)
 *   host_ptr: CPU 端写入 PCL 数据 (uncached, 直达物理内存)
 *   dev_ptr:  GPU 端指针 (传给 kernel, DMA 直读同一物理内存)
 *   无 cache flush/invalidation 开销
 */

#include "semantic_vslam/cuda_voxel_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_vslam {

// 持久 zero-copy 缓冲区 (进程生命周期, 只增不减)
static VoxelPoint* g_h_input  = nullptr;  // CPU 端 (host)
static VoxelPoint* g_d_input  = nullptr;  // GPU 端 (device mapped)
static VoxelPoint* g_h_output = nullptr;
static VoxelPoint* g_d_output = nullptr;
static int g_zc_capacity = 0;

static void ensureZeroCopyBuffers(int N) {
  if (N <= g_zc_capacity) return;

  int new_cap = std::max(N, static_cast<int>(g_zc_capacity * 1.5));

  // 释放旧的
  if (g_h_input)  cudaVoxelGridFreeZeroCopy(g_h_input);
  if (g_h_output) cudaVoxelGridFreeZeroCopy(g_h_output);
  g_h_input = g_d_input = g_h_output = g_d_output = nullptr;

  // 分配新的 zero-copy 缓冲区
  if (!cudaVoxelGridAllocZeroCopy(new_cap, &g_h_input, &g_d_input) ||
      !cudaVoxelGridAllocZeroCopy(new_cap, &g_h_output, &g_d_output)) {
    fprintf(stderr, "[CUDA] Failed to allocate zero-copy buffers\n");
    g_zc_capacity = 0;
    return;
  }
  g_zc_capacity = new_cap;
}

void cudaVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZRGB> &input,
                         pcl::PointCloud<pcl::PointXYZRGB> &output,
                         float voxel_size) {
  const int N = static_cast<int>(input.size());
  if (N == 0)
    return;
  if (voxel_size <= 0.0f) {
    output = input;
    return;
  }

  ensureZeroCopyBuffers(N);
  if (g_zc_capacity < N) return;  // 分配失败

  // ---- PCL → VoxelPoint (写到 CPU 端 zero-copy 内存, uncached 直达物理内存) ----
  for (int i = 0; i < N; ++i) {
    const auto &pt = input.points[i];
    g_h_input[i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};
  }

  // ---- 调用 CUDA (传 GPU 端指针, DMA 直读, 无 H2D 拷贝) ----
  int num_out = cudaVoxelGridFilterRaw(
      g_d_input, N, g_d_output, N, voxel_size);

  // ---- VoxelPoint → PCL (从 CPU 端 zero-copy 内存直读, 无 D2H 拷贝) ----
  output.resize(num_out);
  output.width = num_out;
  output.height = 1;
  output.is_dense = true;

  for (int i = 0; i < num_out; ++i) {
    auto &op = output.points[i];
    op.x = g_h_output[i].x;
    op.y = g_h_output[i].y;
    op.z = g_h_output[i].z;
    op.r = g_h_output[i].r;
    op.g = g_h_output[i].g;
    op.b = g_h_output[i].b;
  }
}

} // namespace semantic_vslam

// ============================================================================
// CudaIncrementalVoxelGrid 实现
// ============================================================================
namespace semantic_vslam {

CudaIncrementalVoxelGrid::CudaIncrementalVoxelGrid(float voxel_size)
    : voxel_size_(voxel_size) {}

void CudaIncrementalVoxelGrid::addCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>& new_cloud) {
  if (new_cloud.empty()) return;

  const int M = static_cast<int>(global_map_.size());
  const int N = static_cast<int>(new_cloud.size());
  const int total = M + N;

  // 复用全局 zero-copy 缓冲区
  ensureZeroCopyBuffers(total);
  if (g_zc_capacity < total) return;

  // ---- 拼接: 先放现有地图 (sort 稳定性下现有点优先) ----
  for (int i = 0; i < M; ++i) {
    const auto& pt = global_map_.points[i];
    g_h_input[i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};
  }
  for (int i = 0; i < N; ++i) {
    const auto& pt = new_cloud.points[i];
    g_h_input[M + i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};
  }

  // ---- CUDA VoxelGrid (GPU 端指针, DMA 直读) ----
  int num_out = cudaVoxelGridFilterRaw(
      g_d_input, total, g_d_output, total, voxel_size_);

  // ---- 更新全局地图 (从 CPU 端 zero-copy 内存直读) ----
  global_map_.resize(num_out);
  global_map_.width = num_out;
  global_map_.height = 1;
  global_map_.is_dense = true;

  for (int i = 0; i < num_out; ++i) {
    auto& op = global_map_.points[i];
    op.x = g_h_output[i].x;
    op.y = g_h_output[i].y;
    op.z = g_h_output[i].z;
    op.r = g_h_output[i].r;
    op.g = g_h_output[i].g;
    op.b = g_h_output[i].b;
  }
}

void CudaIncrementalVoxelGrid::clear() {
  global_map_.clear();
}

} // namespace semantic_vslam
