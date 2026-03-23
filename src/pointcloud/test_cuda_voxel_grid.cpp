/**
 * test_cuda_voxel_grid.cpp
 *
 * CUDA VoxelGrid 单元测试
 *
 * 1. 随机生成 N 个 PointXYZRGB 点 (含语义颜色)
 * 2. 用 CUDA VoxelGrid 下采样
 * 3. 用 PCL VoxelGrid 下采样 (baseline)
 * 4. 对比: 输出点数、语义颜色保留、耗时
 */

#include "semantic_vslam/cuda_voxel_grid.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <random>

// YOLO 语义颜色 (椅子=红, 桌子=蓝, 沙发=绿, ...)
static const uint8_t SEMANTIC_COLORS[][3] = {
    {255, 0, 0},    // 椅子 red
    {0, 0, 255},    // 桌子 blue
    {0, 255, 0},    // 沙发 green
    {255, 255, 0},  // 显示器 yellow
    {255, 0, 255},  // 植物 magenta
};

int main(int argc, char** argv) {
  // ---- 参数 ----
  int total_points = 500000;     // 50 万默认
  float voxel_size = 0.02f;      // 2cm
  int num_repeats = 5;

  if (argc >= 2) total_points = std::atoi(argv[1]);
  if (argc >= 3) voxel_size = std::atof(argv[2]);

  printf("=== CUDA VoxelGrid Unit Test ===\n");
  printf("Points: %d, VoxelSize: %.3f m\n\n", total_points, voxel_size);

  // ---- 生成测试点云 ----
  // 模拟场景: 5m x 5m x 3m 空间, 5 种语义类别的物体
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  cloud->reserve(total_points);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> x_dist(-2.5f, 2.5f);
  std::uniform_real_distribution<float> y_dist(-2.5f, 2.5f);
  std::uniform_real_distribution<float> z_dist(0.0f, 3.0f);
  std::uniform_int_distribution<int> class_dist(0, 4);

  for (int i = 0; i < total_points; ++i) {
    pcl::PointXYZRGB pt;
    pt.x = x_dist(rng);
    pt.y = y_dist(rng);
    pt.z = z_dist(rng);
    int cls = class_dist(rng);
    pt.r = SEMANTIC_COLORS[cls][0];
    pt.g = SEMANTIC_COLORS[cls][1];
    pt.b = SEMANTIC_COLORS[cls][2];
    cloud->push_back(pt);
  }
  cloud->width = total_points;
  cloud->height = 1;
  cloud->is_dense = true;

  // ---- 1. PCL VoxelGrid (baseline) ----
  printf("[PCL VoxelGrid]\n");
  pcl::PointCloud<pcl::PointXYZRGB> pcl_output;
  double pcl_total_ms = 0;

  for (int r = 0; r < num_repeats; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();

    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(pcl_output);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    pcl_total_ms += ms;
    printf("  Run %d: %.1f ms, output %zu points\n", r + 1, ms, pcl_output.size());
  }
  printf("  Avg: %.1f ms, output %zu points\n\n",
         pcl_total_ms / num_repeats, pcl_output.size());

  // ---- 2. CUDA VoxelGrid ----
  printf("[CUDA VoxelGrid]\n");
  pcl::PointCloud<pcl::PointXYZRGB> cuda_output;
  double cuda_total_ms = 0;

  for (int r = 0; r < num_repeats; ++r) {
    cuda_output.clear();

    auto t0 = std::chrono::high_resolution_clock::now();
    semantic_vslam::cudaVoxelGridFilter(*cloud, cuda_output, voxel_size);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    cuda_total_ms += ms;
    printf("  Run %d: %.1f ms, output %zu points\n", r + 1, ms, cuda_output.size());
  }
  printf("  Avg: %.1f ms, output %zu points\n\n",
         cuda_total_ms / num_repeats, cuda_output.size());

  // ---- 3. 对比分析 ----
  printf("=== Comparison ===\n");
  double speedup = (pcl_total_ms / num_repeats) / (cuda_total_ms / num_repeats);
  printf("Speedup: %.1fx\n", speedup);

  // 点数对比 (允许 10% 误差, 因为 PCL 用质心而 CUDA 用 first-point)
  double ratio = static_cast<double>(cuda_output.size()) / pcl_output.size();
  printf("Point count ratio (CUDA/PCL): %.3f\n", ratio);
  if (std::abs(ratio - 1.0) < 0.10) {
    printf("  ✅ Point count match (within 10%%)\n");
  } else {
    printf("  ⚠️  Point count divergence > 10%%\n");
  }

  // 语义颜色验证: 每个输出点的 RGB 必须是合法的语义颜色 (不能是平均后的奇怪颜色)
  int invalid_colors = 0;
  for (const auto& pt : cuda_output.points) {
    bool valid = false;
    for (const auto& sc : SEMANTIC_COLORS) {
      if (pt.r == sc[0] && pt.g == sc[1] && pt.b == sc[2]) {
        valid = true;
        break;
      }
    }
    if (!valid) invalid_colors++;
  }
  printf("Invalid semantic colors: %d / %zu\n", invalid_colors, cuda_output.size());
  if (invalid_colors == 0) {
    printf("  ✅ All colors are valid semantic labels (no averaging)\n");
  } else {
    printf("  ❌ Some colors were averaged/corrupted!\n");
  }

  // ---- 4. 总结 ----
  printf("\n=== Summary ===\n");
  printf("PCL:  %.1f ms (%zu pts)\n", pcl_total_ms / num_repeats, pcl_output.size());
  printf("CUDA: %.1f ms (%zu pts)\n", cuda_total_ms / num_repeats, cuda_output.size());
  printf("Speedup: %.1fx\n", speedup);

  bool pass = (std::abs(ratio - 1.0) < 0.10) && (invalid_colors == 0) && (speedup > 1.0);
  printf("\nResult: %s\n", pass ? "✅ PASS" : "❌ FAIL");

  return pass ? 0 : 1;
}
