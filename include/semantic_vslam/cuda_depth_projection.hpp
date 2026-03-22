#pragma once
/**
 * cuda_depth_projection.hpp
 *
 * GPU 加速的深度→3D 投影 + 语义着色 (融合 kernel)
 *
 * 替代 semantic_cloud_node::generateSemanticCloud 的逐像素 CPU 循环。
 * 640×480 → <1ms kernel + ~2ms memcpy (vs CPU 6-10ms)
 *
 * 注意: 不包含 PCL/Eigen (保持 nvcc 兼容)
 */

#include <cstdint>

namespace semantic_vslam {
namespace cuda {

/**
 * 初始化 GPU 常量表 (调用一次)
 */
void gpuDepthProjectionInit(
    const uint8_t semantic_colors[80][3],
    const bool is_dynamic[80]);

/**
 * GPU 深度投影 + 语义着色 (紧凑 AoS 输出)
 *
 * @param out_packed  预分配 width*height × 16 bytes 的输出缓冲区
 *                    每点: struct { float x, y, z; uint8_t r, g, b, a; }
 *                    调用端展开到 PCL PointXYZRGB
 */
void gpuDepthProjectionRaw(
    const void* depth_data, bool is_16u,
    const uint8_t* label_data,
    const uint8_t* bgr_data,
    int width, int height,
    float fx, float fy, float cx, float cy,
    float depth_scale,
    void* out_packed,
    void* stream = nullptr);

} // namespace cuda
} // namespace semantic_vslam
