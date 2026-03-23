/**
 * cuda_voxel_grid.cu
 *
 * GPU 加速 VoxelGrid 下采样 (底层 CUDA 实现)
 *
 * 算法: calcVoxelIndex → thrust::sort_by_key → markBoundary → inclusive_scan → gatherFirstPoint
 * 颜色策略: first-point (语义颜色不可平均)
 *
 * 注意: 此文件仅包含 CUDA Runtime / Thrust, 不包含 PCL / Eigen
 *       PCL 封装在 cuda_voxel_grid_wrapper.cpp 中
 */

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace semantic_vslam {

// 与 header 中的 VoxelPoint 完全一致 (定义在 namespace 内)
struct VoxelPoint {
  float x, y, z;
  uint8_t r, g, b, pad;
};

// ---- Kernel 1: 计算每个点的 voxel 1D 索引 ----
__global__ void calcVoxelIndexKernel(
    const VoxelPoint* __restrict__ points,
    int num_points,
    float inv_voxel,
    int min_vx, int min_vy, int min_vz,
    int nx, int ny,
    uint32_t* __restrict__ voxel_indices,
    uint32_t* __restrict__ point_indices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  const VoxelPoint& p = points[idx];
  int vx = static_cast<int>(floorf(p.x * inv_voxel)) - min_vx;
  int vy = static_cast<int>(floorf(p.y * inv_voxel)) - min_vy;
  int vz = static_cast<int>(floorf(p.z * inv_voxel)) - min_vz;

  voxel_indices[idx] = static_cast<uint32_t>(vx + nx * vy + nx * ny * vz);
  point_indices[idx] = static_cast<uint32_t>(idx);
}

// ---- Kernel 2: 标记 voxel 边界 (排序后相邻不同 → 1) ----
__global__ void markBoundaryKernel(
    const uint32_t* __restrict__ sorted_voxel_indices,
    int num_points,
    uint32_t* __restrict__ boundary)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  if (idx == 0) {
    boundary[idx] = 1;
  } else {
    boundary[idx] = (sorted_voxel_indices[idx] != sorted_voxel_indices[idx - 1]) ? 1 : 0;
  }
}

// ---- Kernel 3: 提取每个 voxel 的第一个点 ----
__global__ void gatherFirstPointKernel(
    const VoxelPoint* __restrict__ points,
    const uint32_t* __restrict__ sorted_point_indices,
    const uint32_t* __restrict__ boundary,
    const uint32_t* __restrict__ prefix_sum,
    int num_points,
    VoxelPoint* __restrict__ output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  if (boundary[idx] == 1) {
    uint32_t out_idx = prefix_sum[idx] - 1;
    uint32_t pt_idx = sorted_point_indices[idx];
    output[out_idx] = points[pt_idx];
  }
}

// ---- 底层接口实现 ----
int cudaVoxelGridFilterRaw(
    const VoxelPoint* h_input, int N,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z)
{
  if (N <= 0) return 0;

  const float inv_voxel = 1.0f / voxel_size;

  int min_vx = static_cast<int>(std::floor(min_x * inv_voxel));
  int min_vy = static_cast<int>(std::floor(min_y * inv_voxel));
  int min_vz = static_cast<int>(std::floor(min_z * inv_voxel));
  int max_vx = static_cast<int>(std::floor(max_x * inv_voxel));
  int max_vy = static_cast<int>(std::floor(max_y * inv_voxel));
  int max_vz = static_cast<int>(std::floor(max_z * inv_voxel));

  int nx = max_vx - min_vx + 1;
  int ny = max_vy - min_vy + 1;

  // ---- 使用 thrust::device_vector 管理 GPU 内存 ----
  // (Thrust 内部管理分配/释放, 避免 raw cudaMalloc 与 Thrust 冲突)
  thrust::device_vector<VoxelPoint> d_points(h_input, h_input + N);
  thrust::device_vector<uint32_t> d_voxel_idx(N);
  thrust::device_vector<uint32_t> d_point_idx(N);
  thrust::device_vector<uint32_t> d_boundary(N);
  thrust::device_vector<uint32_t> d_prefix(N);

  // ---- 计算 voxel 索引 ----
  const int BLOCK = 256;
  const int GRID = (N + BLOCK - 1) / BLOCK;

  calcVoxelIndexKernel<<<GRID, BLOCK>>>(
      thrust::raw_pointer_cast(d_points.data()), N, inv_voxel,
      min_vx, min_vy, min_vz, nx, ny,
      thrust::raw_pointer_cast(d_voxel_idx.data()),
      thrust::raw_pointer_cast(d_point_idx.data()));
  cudaDeviceSynchronize();

  // ---- Thrust sort_by_key (in-place) ----
  thrust::sort_by_key(d_voxel_idx.begin(), d_voxel_idx.end(), d_point_idx.begin());

  // ---- 标记边界 + inclusive scan ----
  markBoundaryKernel<<<GRID, BLOCK>>>(
      thrust::raw_pointer_cast(d_voxel_idx.data()), N,
      thrust::raw_pointer_cast(d_boundary.data()));
  cudaDeviceSynchronize();

  thrust::inclusive_scan(d_boundary.begin(), d_boundary.end(), d_prefix.begin());

  // 取出 unique voxel 总数
  uint32_t num_unique = d_prefix.back();  // 自动 D2H 拷贝

  if (num_unique == 0 || static_cast<int>(num_unique) > max_output) {
    return 0;
  }

  // ---- 提取每个 voxel 第一个点 ----
  thrust::device_vector<VoxelPoint> d_output(num_unique);

  gatherFirstPointKernel<<<GRID, BLOCK>>>(
      thrust::raw_pointer_cast(d_points.data()),
      thrust::raw_pointer_cast(d_point_idx.data()),
      thrust::raw_pointer_cast(d_boundary.data()),
      thrust::raw_pointer_cast(d_prefix.data()),
      N,
      thrust::raw_pointer_cast(d_output.data()));
  cudaDeviceSynchronize();

  // ---- D2H 下载 ----
  thrust::copy(d_output.begin(), d_output.end(), h_output);

  return static_cast<int>(num_unique);
}

}  // namespace semantic_vslam
