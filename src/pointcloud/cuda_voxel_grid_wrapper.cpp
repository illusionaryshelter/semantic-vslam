/**
 * cuda_voxel_grid_wrapper.cpp
 *
 * PCL ↔ VoxelPoint 转换封装
 * 此文件由 g++ 编译 (可用 PCL/Eigen), 调用 .cu 中的 CUDA 接口
 */

#include "semantic_vslam/cuda_voxel_grid.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

namespace semantic_vslam {

void cudaVoxelGridFilter(
    const pcl::PointCloud<pcl::PointXYZRGB>& input,
    pcl::PointCloud<pcl::PointXYZRGB>& output,
    float voxel_size)
{
  const int N = static_cast<int>(input.size());
  if (N == 0) return;
  if (voxel_size <= 0.0f) {
    output = input;
    return;
  }

  // ---- PCL → VoxelPoint 打包 + CPU 端 min/max 计算 ----
  std::vector<VoxelPoint> h_input(N);
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = min_x, max_y = max_x;
  float min_z = min_x, max_z = max_x;

  for (int i = 0; i < N; ++i) {
    const auto& pt = input.points[i];
    h_input[i] = {pt.x, pt.y, pt.z, pt.r, pt.g, pt.b, 0};

    if (pt.x < min_x) min_x = pt.x;
    if (pt.x > max_x) max_x = pt.x;
    if (pt.y < min_y) min_y = pt.y;
    if (pt.y > max_y) max_y = pt.y;
    if (pt.z < min_z) min_z = pt.z;
    if (pt.z > max_z) max_z = pt.z;
  }

  // ---- 调用 CUDA 底层 ----
  std::vector<VoxelPoint> h_output(N);  // 最坏情况: 每点一个 voxel
  int num_out = cudaVoxelGridFilterRaw(
      h_input.data(), N,
      h_output.data(), N,
      voxel_size,
      min_x, max_x, min_y, max_y, min_z, max_z);

  // ---- VoxelPoint → PCL 转换 ----
  output.resize(num_out);
  output.width = num_out;
  output.height = 1;
  output.is_dense = true;

  for (int i = 0; i < num_out; ++i) {
    auto& op = output.points[i];
    op.x = h_output[i].x;
    op.y = h_output[i].y;
    op.z = h_output[i].z;
    op.r = h_output[i].r;
    op.g = h_output[i].g;
    op.b = h_output[i].b;
  }
}

}  // namespace semantic_vslam
