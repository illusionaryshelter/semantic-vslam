# Semantic VSLAM Project

## 项目概述
本项目是一个纯视觉 SLAM 系统，使用ROS2作为统一接口，全程使用 C++。
**核心流程**：RGB 图像输入 -> YOLOv8-seg 分割提取掩码 -> 结合深度/双目信息生成带有语义标签的自定义点云 -> 输入给 RTAB-Map 进行配准和建图。
**目标硬件**：Jetson Orin Nano 8GB（请在编写 YOLOv8-seg 推理代码时，优先考虑使用 TensorRT 以获得实时的端侧性能）。

## 技术栈与依赖
- C++17, CMake
- 核心算法：RTAB-Map, PCL (Point Cloud Library)
- 图像与推理：OpenCV, TensorRT / ONNX Runtime

## 调试与开发规则（严格遵守）
1. **接口友好**： 无需使用复杂的设计模式，因为用户比较熟悉c++，而是充分考虑性能以及良好的注释
2. **防御性编程**：在 OpenCV 矩阵操作、模型加载和点云坐标转换时，必须加入完善的 `try-catch` 块和空指针检查，拒绝静默失败（Silent Failures）。
3. **构建可视化**：在生成 `CMakeLists.txt` 时，默认开启 `set(CMAKE_VERBOSE_MAKEFILE ON)`，确保在编译报错时能看到完整的链接信息，方便调试。
4. **渐进式开发**：禁止一次性生成整个项目的代码。必须按模块推进（例如：先只写 YOLOv8-seg 的推理与掩码可视化，测试通过后再写点云转换，最后接入 RTAB-Map）。


