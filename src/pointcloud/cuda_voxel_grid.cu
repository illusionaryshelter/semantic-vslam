/**
 * cuda_voxel_grid.cu
 *
 * GPU 加速 VoxelGrid 下采样 (底层 CUDA 实现)
 *
 * 算法: calcVoxelIndex → thrust::sort_by_key → markBoundary → inclusive_scan → gatherFirstPoint
 * 颜色策略: first-point (语义颜色不可平均)
 *
 * 空间哈希: 使用 uint64_t bit-packing (21位/轴, 无碰撞)
 *   - 替代了原来的 dense grid index (uint32_t, 大地图溢出)
 *   - 每轴 ±1,048,575 个 voxel = ±20km @ 0.02m
 *   - 排序后相同 voxel 的点必然相邻, 无哈希碰撞
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

// ---- 空间哈希常量 ----
// 每轴 21 位, 总共 63 位 (uint64_t 的低 63 位)
// 偏移量: 将有符号 voxel 坐标映射到无符号区间
// 范围: ±1,048,575 个 voxel = ±20,971m @ 0.02m 体素
static constexpr uint64_t VOXEL_BITS = 21;
static constexpr uint64_t VOXEL_OFFSET = (1ULL << (VOXEL_BITS - 1)); // 1048576

// ---- Kernel 1: 计算每个点的 voxel key (uint64_t bit-packing) ----
// 对比旧版 dense grid: vx + nx*vy + nx*ny*vz (uint32_t, 大地图溢出)
// 新版: 3×21-bit 无碰撞编码, 无需 bounding box
__global__ void calcVoxelIndexKernel(
    const VoxelPoint* __restrict__ points,
    int num_points,
    float inv_voxel,
    uint64_t* __restrict__ voxel_keys,
    uint32_t* __restrict__ point_indices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  const VoxelPoint& p = points[idx];

  // 绝对 voxel 坐标 (有符号)
  int64_t vx = static_cast<int64_t>(floorf(p.x * inv_voxel));
  int64_t vy = static_cast<int64_t>(floorf(p.y * inv_voxel));
  int64_t vz = static_cast<int64_t>(floorf(p.z * inv_voxel));

  // bit-packing: [vx+offset : 21bit] [vy+offset : 21bit] [vz+offset : 21bit]
  // 偏移保证无符号, 排序后相同 voxel 的点必然相邻
  uint64_t ux = static_cast<uint64_t>(vx + static_cast<int64_t>(VOXEL_OFFSET));
  uint64_t uy = static_cast<uint64_t>(vy + static_cast<int64_t>(VOXEL_OFFSET));
  uint64_t uz = static_cast<uint64_t>(vz + static_cast<int64_t>(VOXEL_OFFSET));

  voxel_keys[idx] = (ux << (VOXEL_BITS * 2)) | (uy << VOXEL_BITS) | uz;
  point_indices[idx] = static_cast<uint32_t>(idx);
}

// ---- Kernel 2: 标记 voxel 边界 (排序后相邻不同 → 1) ----
__global__ void markBoundaryKernel(
    const uint64_t* __restrict__ sorted_voxel_keys,
    int num_points,
    uint32_t* __restrict__ boundary)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  if (idx == 0) {
    boundary[idx] = 1;
  } else {
    boundary[idx] = (sorted_voxel_keys[idx] != sorted_voxel_keys[idx - 1]) ? 1 : 0;
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

// ---- 底层接口实现 (简化: 不再需要 bounding box 参数) ----
int cudaVoxelGridFilterRaw(
    const VoxelPoint* h_input, int N,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float /*min_x*/, float /*max_x*/,
    float /*min_y*/, float /*max_y*/,
    float /*min_z*/, float /*max_z*/)
{
  if (N <= 0) return 0;

  const float inv_voxel = 1.0f / voxel_size;

  // ---- 使用 thrust::device_vector 管理 GPU 内存 ----
  thrust::device_vector<VoxelPoint> d_points(h_input, h_input + N);
  thrust::device_vector<uint64_t>   d_voxel_keys(N);
  thrust::device_vector<uint32_t>   d_point_idx(N);
  thrust::device_vector<uint32_t>   d_boundary(N);
  thrust::device_vector<uint32_t>   d_prefix(N);

  // ---- 计算 voxel key (uint64_t bit-packing, 无溢出) ----
  const int BLOCK = 256;
  const int GRID = (N + BLOCK - 1) / BLOCK;

  calcVoxelIndexKernel<<<GRID, BLOCK>>>(
      thrust::raw_pointer_cast(d_points.data()), N, inv_voxel,
      thrust::raw_pointer_cast(d_voxel_keys.data()),
      thrust::raw_pointer_cast(d_point_idx.data()));
  cudaDeviceSynchronize();

  // ---- Thrust sort_by_key (uint64_t key, 比 uint32_t 稍慢但安全) ----
  thrust::sort_by_key(d_voxel_keys.begin(), d_voxel_keys.end(), d_point_idx.begin());

  // ---- 标记边界 + inclusive scan ----
  markBoundaryKernel<<<GRID, BLOCK>>>(
      thrust::raw_pointer_cast(d_voxel_keys.data()), N,
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
