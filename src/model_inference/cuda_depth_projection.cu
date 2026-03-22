/**
 * cuda_depth_projection.cu — v3
 *
 * GPU 加速的深度→3D 投影 + 语义着色 (融合 kernel)
 *
 * v3 优化: 输出 32-byte PCL-compatible 布局
 *   调用端: 单次 memcpy 直接填充 pcl::PointXYZRGB 内存
 *   消除了: 中间缓冲区分配 + per-point 转换循环
 *
 * PCL PointXYZRGB 布局 (32 bytes):
 *   [0..3]   float x
 *   [4..7]   float y
 *   [8..11]  float z
 *   [12..15] float pad (data[3])
 *   [16]     uint8_t b  ← 注意 BGR 顺序!
 *   [17]     uint8_t g
 *   [18]     uint8_t r
 *   [19]     uint8_t a
 *   [20..31] padding
 */

#include "semantic_vslam/cuda_depth_projection.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace semantic_vslam {
namespace cuda {

__constant__ uint8_t d_semantic_colors[80][3];  // RGB 顺序
__constant__ uint8_t d_is_dynamic[80];

// PCL PointXYZRGB 内存布局 (32 bytes, 与 PCL 二进制兼容)
struct alignas(16) PCLPointXYZRGB {
  float x, y, z, pad0;          // 16 bytes (SSE-aligned)
  uint8_t b, g, r, a;           // 4 bytes (注意: BGR 顺序!)
  float pad1, pad2, pad3;       // 12 bytes padding → total 32
};

// ======================================================================
// 融合 Kernel v3: 直接输出 PCL 兼容布局
// ======================================================================
__global__ void kernelDepthProjection(
    const uint16_t* __restrict__ depth_u16,
    const float*    __restrict__ depth_f32,
    const uint8_t*  __restrict__ label_map,
    const uint8_t*  __restrict__ bgr,
    PCLPointXYZRGB* __restrict__ out,
    int width, int height,
    float inv_fx, float inv_fy, float cx, float cy,
    float depth_scale)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height) return;

  int u = idx % width;
  int v = idx / width;

  // 1. 读取深度
  float z;
  if (depth_u16) {
    z = static_cast<float>(depth_u16[idx]) * depth_scale;
  } else {
    z = depth_f32[idx];
  }

  PCLPointXYZRGB pt;
  pt.pad0 = 1.0f;  // PCL data[3] 约定
  pt.a = 255;
  pt.pad1 = pt.pad2 = pt.pad3 = 0.0f;

  // 2. 有效性检查
  if (z <= 0.01f || z > 10.0f || isnan(z)) {
    pt.x = nanf(""); pt.y = nanf(""); pt.z = nanf("");
    pt.b = 0; pt.g = 0; pt.r = 0;
    out[idx] = pt;
    return;
  }

  // 3. 动态物体检查
  uint8_t lbl = label_map[idx];
  if (lbl > 0 && (lbl - 1) < 80 && d_is_dynamic[lbl - 1]) {
    pt.x = nanf(""); pt.y = nanf(""); pt.z = nanf("");
    pt.b = 0; pt.g = 0; pt.r = 0;
    out[idx] = pt;
    return;
  }

  // 4. 深度 → 3D
  pt.x = (static_cast<float>(u) - cx) * z * inv_fx;
  pt.y = (static_cast<float>(v) - cy) * z * inv_fy;
  pt.z = z;

  // 5. 颜色 (注意: PCL 布局是 BGR!)
  if (lbl > 0 && (lbl - 1) < 80) {
    int cls = lbl - 1;
    pt.r = d_semantic_colors[cls][0];  // 常量表是 RGB
    pt.g = d_semantic_colors[cls][1];
    pt.b = d_semantic_colors[cls][2];
  } else {
    // 输入是 BGR, PCL 也是 BGR → 直接复制
    int off = idx * 3;
    pt.b = bgr[off + 0];
    pt.g = bgr[off + 1];
    pt.r = bgr[off + 2];
  }

  out[idx] = pt;
}

// ======================================================================
// Zero-copy 缓冲区
// ======================================================================
static struct DepthProjBuffers {
  void*           h_depth = nullptr;  void*           d_depth = nullptr;
  uint8_t*        h_label = nullptr;  uint8_t*        d_label = nullptr;
  uint8_t*        h_rgb   = nullptr;  uint8_t*        d_rgb   = nullptr;
  PCLPointXYZRGB* h_out   = nullptr;  PCLPointXYZRGB* d_out   = nullptr;
  int alloc_pixels = 0;

  void ensure(int pixels) {
    if (alloc_pixels >= pixels) return;
    release();
    cudaHostAlloc(&h_depth, pixels * sizeof(float), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_depth, h_depth, 0);
    cudaHostAlloc(&h_label, pixels, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_label, h_label, 0);
    cudaHostAlloc(&h_rgb, pixels * 3, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_rgb, h_rgb, 0);
    cudaHostAlloc(&h_out, pixels * sizeof(PCLPointXYZRGB), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_out, h_out, 0);
    alloc_pixels = pixels;
  }

  void release() {
    if (h_depth) { cudaFreeHost(h_depth); h_depth = nullptr; }
    if (h_label) { cudaFreeHost(h_label); h_label = nullptr; }
    if (h_rgb)   { cudaFreeHost(h_rgb);   h_rgb = nullptr; }
    if (h_out)   { cudaFreeHost(h_out);   h_out = nullptr; }
    d_depth = d_label = nullptr; d_rgb = nullptr; d_out = nullptr;
    alloc_pixels = 0;
  }
  ~DepthProjBuffers() { release(); }
} g_dp_bufs;

static bool g_tables_initialized = false;

void gpuDepthProjectionInit(
    const uint8_t semantic_colors[80][3],
    const bool is_dynamic[80])
{
  cudaMemcpyToSymbol(d_semantic_colors, semantic_colors, 80 * 3);
  uint8_t flags[80];
  for (int i = 0; i < 80; ++i) flags[i] = is_dynamic[i] ? 1 : 0;
  cudaMemcpyToSymbol(d_is_dynamic, flags, 80);
  g_tables_initialized = true;
}

// ======================================================================
// Public API: 深度投影 + 语义着色
//
// out_pcl_data: 调用端 cloud.points.data() (pcl::PointXYZRGB*, 32 bytes each)
//              直接大 memcpy 填充, 零转换循环
// ======================================================================
void gpuDepthProjectionRaw(
    const void* depth_data, bool is_16u,
    const uint8_t* label_data,
    const uint8_t* bgr_data,
    int width, int height,
    float fx, float fy, float cx, float cy,
    float depth_scale,
    void* out_pcl_data,
    void* stream)
{
  if (!g_tables_initialized) {
    fprintf(stderr, "[cuda] ERROR: gpuDepthProjectionInit() not called!\n");
    return;
  }

  int pixels = width * height;
  if (pixels <= 0) return;
  g_dp_bufs.ensure(pixels);

  // 拷贝输入到 zero-copy
  if (is_16u) {
    memcpy(g_dp_bufs.h_depth, depth_data, pixels * sizeof(uint16_t));
  } else {
    memcpy(g_dp_bufs.h_depth, depth_data, pixels * sizeof(float));
  }
  memcpy(g_dp_bufs.h_label, label_data, pixels);
  memcpy(g_dp_bufs.h_rgb, bgr_data, pixels * 3);

  float inv_fx = 1.0f / fx;
  float inv_fy = 1.0f / fy;

  const int threads = 256;
  const int blocks = (pixels + threads - 1) / threads;
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  uint16_t* d_u16 = is_16u ? static_cast<uint16_t*>(g_dp_bufs.d_depth) : nullptr;
  float*    d_f32 = is_16u ? nullptr : static_cast<float*>(g_dp_bufs.d_depth);

  kernelDepthProjection<<<blocks, threads, 0, s>>>(
      d_u16, d_f32,
      g_dp_bufs.d_label, g_dp_bufs.d_rgb,
      g_dp_bufs.d_out,
      width, height, inv_fx, inv_fy, cx, cy, depth_scale);

  if (s) cudaStreamSynchronize(s);
  else   cudaDeviceSynchronize();

  // 单次 memcpy: zero-copy output → PCL cloud memory (307K × 32 = ~9.4MB)
  memcpy(out_pcl_data, g_dp_bufs.h_out, pixels * sizeof(PCLPointXYZRGB));
}

} // namespace cuda
} // namespace semantic_vslam
