/**
 * cuda_voxel_grid.cu
 *
 * GPU 加速 VoxelGrid 下采样 (底层 CUDA 实现)
 *
 * 算法: calcVoxelIndex → thrust::sort_by_key → markBoundary → inclusive_scan → gatherFirstPoint
 * 颜色策略: first-point (语义颜色不可平均)
 *
 * 空间哈希: uint64_t bit-packing (21位/轴, 无碰撞, ±20km @ 0.02m)
 *
 * 内存策略 (Jetson UMA):
 *   1. GPUPool 中间缓冲区: cudaMalloc (纯 device memory, GPU L2 cache 正常)
 *      - voxel_keys, point_idx, boundary, prefix 仅 GPU 读写
 *      - CPU 仅读 prefix[N-1] → cudaMemcpy D2H 4 字节
 *   2. I/O 缓冲区: cudaHostAllocMapped (pinned + uncached = 真零拷贝)
 *      - CPU 写入直达物理内存, GPU DMA 直读, 无 cache flush 开销
 *      - 对比 cudaMallocManaged: 后者 CPU cacheable, 需要 cache flush/invalidation
 *   3. 持久分配: 一次分配跨调用复用, 1.5x 增长, 避免反复 alloc/free
 *
 * 注意: 此文件仅包含 CUDA Runtime / Thrust, 不包含 PCL / Eigen
 *       PCL 封装在 cuda_voxel_grid_wrapper.cpp 中
 */

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <algorithm>

namespace semantic_vslam {

// 与 header 中的 VoxelPoint 完全一致 (定义在 namespace 内)
struct VoxelPoint {
  float x, y, z;
  uint8_t r, g, b, pad;
};

// ---- 空间哈希常量 ----
static constexpr uint64_t VOXEL_BITS = 21;
static constexpr uint64_t VOXEL_OFFSET = (1ULL << (VOXEL_BITS - 1));

// ============================================================================
// 持久 GPU 内存池: cudaMalloc (纯 device memory)
//
// 中间缓冲区仅 GPU 读写: sort, scan, boundary, gather
// 纯 device memory 保证 GPU L2 cache 正常工作
// (对比 cudaMallocManaged: GPU 也走 cache coherency protocol → 额外开销)
// (对比 cudaHostAllocMapped: GPU cache 被 bypass → sort 随机访问极慢)
// ============================================================================
struct GPUPool {
  uint64_t*   voxel_keys = nullptr;  // Voxel hash keys (sort 随机访问, 必须 device)
  uint32_t*   point_idx  = nullptr;  // 原始点索引 (随 key 一起 sort, 必须 device)
  uint32_t*   boundary   = nullptr;  // Voxel 边界标记
  uint32_t*   prefix     = nullptr;  // Inclusive scan 前缀和
  int capacity = 0;

  void ensure(int N) {
    if (N <= capacity) return;
    int new_cap = std::max(N, static_cast<int>(capacity * 1.5));
    free_all();

    cudaMalloc(&voxel_keys, new_cap * sizeof(uint64_t));
    cudaMalloc(&point_idx,  new_cap * sizeof(uint32_t));
    cudaMalloc(&boundary,   new_cap * sizeof(uint32_t));
    cudaMalloc(&prefix,     new_cap * sizeof(uint32_t));
    capacity = new_cap;
  }

  void free_all() {
    if (voxel_keys) { cudaFree(voxel_keys); voxel_keys = nullptr; }
    if (point_idx)  { cudaFree(point_idx);  point_idx  = nullptr; }
    if (boundary)   { cudaFree(boundary);   boundary   = nullptr; }
    if (prefix)     { cudaFree(prefix);     prefix     = nullptr; }
    capacity = 0;
  }

  ~GPUPool() { free_all(); }
};

static GPUPool g_pool;

// ---- Kernel 1: 计算每个点的 voxel key (uint64_t bit-packing) ----
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

  int64_t vx = static_cast<int64_t>(floorf(p.x * inv_voxel));
  int64_t vy = static_cast<int64_t>(floorf(p.y * inv_voxel));
  int64_t vz = static_cast<int64_t>(floorf(p.z * inv_voxel));

  uint64_t ux = static_cast<uint64_t>(vx + static_cast<int64_t>(VOXEL_OFFSET));
  uint64_t uy = static_cast<uint64_t>(vy + static_cast<int64_t>(VOXEL_OFFSET));
  uint64_t uz = static_cast<uint64_t>(vz + static_cast<int64_t>(VOXEL_OFFSET));

  voxel_keys[idx] = (ux << (VOXEL_BITS * 2)) | (uy << VOXEL_BITS) | uz;
  point_indices[idx] = static_cast<uint32_t>(idx);
}

// ---- Kernel 2: 标记 voxel 边界 ----
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

// ============================================================================
// 底层接口
//
// input/h_output 可以是:
//   - cudaHostAllocMapped 指针 (真零拷贝, wrapper 使用)
//   - 普通 host 指针 (单元测试, 需要 H2D/D2H → 由调用者自行处理)
//   - device 指针 (直接 GPU 使用)
// ============================================================================
int cudaVoxelGridFilterRaw(
    const VoxelPoint* input, int N,
    VoxelPoint* h_output, int max_output,
    float voxel_size,
    float /*min_x*/, float /*max_x*/,
    float /*min_y*/, float /*max_y*/,
    float /*min_z*/, float /*max_z*/)
{
  if (N <= 0) return 0;

  const float inv_voxel = 1.0f / voxel_size;

  g_pool.ensure(N);

  const int BLOCK = 256;
  const int GRID = (N + BLOCK - 1) / BLOCK;

  // ---- 计算 voxel key (input: mapped/device, output: device) ----
  calcVoxelIndexKernel<<<GRID, BLOCK>>>(
      input, N, inv_voxel,
      g_pool.voxel_keys, g_pool.point_idx);
  cudaDeviceSynchronize();

  // ---- Thrust sort (纯 device memory, GPU L2 cached) ----
  thrust::device_ptr<uint64_t> keys_begin(g_pool.voxel_keys);
  thrust::device_ptr<uint32_t> vals_begin(g_pool.point_idx);
  thrust::sort_by_key(keys_begin, keys_begin + N, vals_begin);

  // ---- 标记边界 + inclusive scan (纯 device memory) ----
  markBoundaryKernel<<<GRID, BLOCK>>>(
      g_pool.voxel_keys, N, g_pool.boundary);
  cudaDeviceSynchronize();

  thrust::device_ptr<uint32_t> bnd_begin(g_pool.boundary);
  thrust::device_ptr<uint32_t> pfx_begin(g_pool.prefix);
  thrust::inclusive_scan(bnd_begin, bnd_begin + N, pfx_begin);

  // ---- 读取 num_unique: cudaMemcpy D2H 4 字节 (prefix 是 device memory) ----
  uint32_t num_unique = 0;
  cudaMemcpy(&num_unique, g_pool.prefix + N - 1,
             sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if (num_unique == 0 || static_cast<int>(num_unique) > max_output) {
    return 0;
  }

  // ---- 提取每个 voxel 第一个点 (直接写到输出缓冲区) ----
  gatherFirstPointKernel<<<GRID, BLOCK>>>(
      input, g_pool.point_idx, g_pool.boundary, g_pool.prefix,
      N, h_output);
  cudaDeviceSynchronize();

  return static_cast<int>(num_unique);
}

// ============================================================================
// Zero-copy 内存分配/释放 (cudaHostAllocMapped)
//
// Jetson UMA 真零拷贝:
//   CPU 写入 → 直达物理内存 (uncached, 无 cache flush)
//   GPU 读取 → DMA 从物理内存直读 (无 cache coherency 开销)
//
// 返回两个指针:
//   host_ptr: CPU 端地址 (用于填充/读取数据)
//   dev_ptr:  GPU 端地址 (传给 CUDA kernel)
//   两者指向同一物理内存
// ============================================================================
bool cudaVoxelGridAllocZeroCopy(int max_points,
                                VoxelPoint** host_ptr,
                                VoxelPoint** dev_ptr) {
  if (!host_ptr || !dev_ptr || max_points <= 0) return false;

  size_t bytes = max_points * sizeof(VoxelPoint);
  cudaError_t err = cudaHostAlloc(host_ptr, bytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA] cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
    return false;
  }

  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(dev_ptr), *host_ptr, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA] cudaHostGetDevicePointer failed: %s\n",
            cudaGetErrorString(err));
    cudaFreeHost(*host_ptr);
    *host_ptr = nullptr;
    return false;
  }

  return true;
}

void cudaVoxelGridFreeZeroCopy(VoxelPoint* host_ptr) {
  if (host_ptr) cudaFreeHost(host_ptr);
}

// ---- 兼容旧 API (保留但标记为 deprecated) ----
VoxelPoint* cudaVoxelGridAllocManaged(int max_points) {
  VoxelPoint* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, max_points * sizeof(VoxelPoint));
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA] cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
    return nullptr;
  }
  return ptr;
}

void cudaVoxelGridFreeManaged(VoxelPoint* ptr) {
  if (ptr) cudaFree(ptr);
}

}  // namespace semantic_vslam
