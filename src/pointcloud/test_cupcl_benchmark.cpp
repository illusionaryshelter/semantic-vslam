/**
 * test_cupcl_benchmark.cpp
 *
 * 对标 NVIDIA cuPCL 官方 benchmark
 * 使用 cuPCL 的 sample.pcd (119,978 XYZ 点), LeafSize = 1.0
 *
 * cuPCL 官方结果 (Jetson Xavier AGX 8GB, CUDA 10.2):
 *   cuPCL VoxelGrid: 3.13 ms → 3440 pts
 *   PCL   VoxelGrid: 7.26 ms → 3440 pts
 */

#include "semantic_vslam/cuda_voxel_grid.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  std::string pcd_path = "/tmp/sample.pcd";
  float leaf_size = 1.0f;
  int num_repeats = 10;

  if (argc >= 2) pcd_path = argv[1];
  if (argc >= 3) leaf_size = std::atof(argv[2]);

  printf("=== cuPCL Benchmark Comparison ===\n");
  printf("PCD: %s, LeafSize: %.1f\n\n", pcd_path.c_str(), leaf_size);

  // ---- 加载 PCD (XYZ only → 转为 XYZRGB) ----
  pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(pcd_path, *xyz_cloud) < 0) {
    fprintf(stderr, "Failed to load %s\n", pcd_path.c_str());
    return 1;
  }
  printf("Loaded %zu XYZ points\n\n", xyz_cloud->size());

  // 转为 XYZRGB (添加固定颜色, 不影响体素计算)
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  cloud->reserve(xyz_cloud->size());
  for (const auto& pt : xyz_cloud->points) {
    pcl::PointXYZRGB rpt;
    rpt.x = pt.x; rpt.y = pt.y; rpt.z = pt.z;
    rpt.r = 255; rpt.g = 0; rpt.b = 0;
    cloud->push_back(rpt);
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = true;

  // ---- 1. PCL VoxelGrid (baseline, 对标 cuPCL 的 PCL 结果) ----
  printf("[PCL VoxelGrid]\n");
  pcl::PointCloud<pcl::PointXYZRGB> pcl_output;
  double pcl_total_ms = 0;
  double pcl_best_ms = 1e9;

  for (int r = 0; r < num_repeats; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();

    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(pcl_output);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    pcl_total_ms += ms;
    if (ms < pcl_best_ms) pcl_best_ms = ms;
    printf("  Run %d: %.2f ms, output %zu pts\n", r + 1, ms, pcl_output.size());
  }
  printf("  Avg: %.2f ms, Best: %.2f ms, output %zu pts\n\n",
         pcl_total_ms / num_repeats, pcl_best_ms, pcl_output.size());

  // ---- 2. Our CUDA VoxelGrid ----
  printf("[Our CUDA VoxelGrid]\n");
  pcl::PointCloud<pcl::PointXYZRGB> cuda_output;
  double cuda_total_ms = 0;
  double cuda_best_ms = 1e9;

  for (int r = 0; r < num_repeats; ++r) {
    cuda_output.clear();

    auto t0 = std::chrono::high_resolution_clock::now();
    semantic_vslam::cudaVoxelGridFilter(*cloud, cuda_output, leaf_size);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    cuda_total_ms += ms;
    if (ms < cuda_best_ms) cuda_best_ms = ms;
    printf("  Run %d: %.2f ms, output %zu pts\n", r + 1, ms, cuda_output.size());
  }
  printf("  Avg: %.2f ms, Best: %.2f ms, output %zu pts\n\n",
         cuda_total_ms / num_repeats, cuda_best_ms, cuda_output.size());

  // ---- 3. cuPCL 官方参考 (from README) ----
  printf("[cuPCL Reference (Xavier AGX, CUDA 10.2)]\n");
  printf("  cuPCL VoxelGrid: 3.13 ms → 3440 pts\n");
  printf("  PCL   VoxelGrid: 7.26 ms → 3440 pts\n\n");

  // ---- 4. 对比分析 ----
  printf("=== Comparison ===\n");
  printf("                    PCL         CUDA        Speedup\n");
  printf("cuPCL (Xavier):     7.26 ms     3.13 ms     2.3x\n");
  printf("Ours  (Orin Nano): %.2f ms    %.2f ms    %.1fx\n",
         pcl_best_ms, cuda_best_ms, pcl_best_ms / cuda_best_ms);

  // 输出点数校验
  printf("\nPoint count check:\n");
  printf("  cuPCL reference: 3440 pts\n");
  printf("  PCL output:      %zu pts\n", pcl_output.size());
  printf("  CUDA output:     %zu pts\n", cuda_output.size());

  if (pcl_output.size() == cuda_output.size()) {
    printf("  ✅ PCL == CUDA point count match\n");
  } else {
    printf("  ⚠️ PCL != CUDA point count (%zu vs %zu)\n",
           pcl_output.size(), cuda_output.size());
  }

  // cuPCL 参考点数对比 (允许版本差异)
  double ref_ratio = static_cast<double>(cuda_output.size()) / 3440.0;
  if (std::abs(ref_ratio - 1.0) < 0.05) {
    printf("  ✅ Matches cuPCL reference (3440 pts)\n");
  } else {
    printf("  ⚠️ Differs from cuPCL reference (expected ~3440, got %zu) - PCL version difference\n",
           cuda_output.size());
  }

  return 0;
}
