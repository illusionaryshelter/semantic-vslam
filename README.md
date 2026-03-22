# 🧠 Semantic VSLAM

> **基于 YOLOv8-seg + RTAB-Map 的实时语义视觉 SLAM 系统**
>
> 目标硬件: NVIDIA Jetson Orin Nano 8GB | C++ / CUDA | ROS 2 Humble

将 YOLOv8 实例分割与 RTAB-Map 视觉 SLAM 融合，在 RGB-D 输入上实时生成**语义着色 3D 点云地图**、**2D 占据栅格地图**和**物体级 3D 包围盒**，为下游导航和场景理解提供语义感知能力。

---

## 📐 系统架构

```
                          ┌──────────────────────────────────────────────┐
                          │          semantic_cloud_node                 │
 ┌─────────────┐          │  ┌──────────┐  ┌─────────────────────────┐  │
 │ Astra Pro / │  RGB     │  │ TensorRT │  │ generateSemanticCloud   │  │
 │ RealSense   │─────────▶│  │ YOLOv8   │─▶│ depth→3D + 语义着色      │──┼──▶ /semantic_cloud
 │ (RGB-D)     │  Depth   │  │ -seg     │  │ + 动态物体过滤           │  │   /label_map
 └──────┬──────┘─────────▶│  └──────────┘  └─────────────────────────┘  │
        │                 └──────────────────────────────────────────────┘
        │                                          │
        │            ┌──────────────────┐          │          ┌──────────────────┐
        │            │ rgbd_odometry    │          ├─────────▶│ semantic_map_node│
        └───────────▶│ (官方 rtabmap)    │          │          │ TF累积 + VoxelGrid│
                     └────────┬─────────┘          │          └────────┬─────────┘
                              │                    │                   │
                     ┌────────▼─────────┐          │          /semantic_map_cloud (3D 语义地图)
                     │ rtabmap SLAM     │          │          /grid_map           (2D 栅格地图)
                     │ (官方 rtabmap)    │          │
                     └──────────────────┘          │          ┌──────────────────┐
                                                   └─────────▶│ object_map_node  │
                                                              │ 3D 聚类 + 包围盒  │
                                                              └────────┬─────────┘
                                                                       │
                                                              /object_markers (MarkerArray)
```

### 输出 Topics

| Topic | 类型 | 说明 |
|---|---|---|
| `/semantic_vslam/semantic_cloud` | PointCloud2 | 单帧语义着色 3D 点云 |
| `/semantic_vslam/semantic_map_cloud` | PointCloud2 | 累积语义着色全局地图 |
| `/semantic_vslam/grid_map` | OccupancyGrid | 语义 2D 占据栅格 |
| `/semantic_vslam/label_map` | Image (CV_8UC1) | 逐像素语义标签图 |
| `/semantic_vslam/object_markers` | MarkerArray | 物体级 3D 包围盒 |

---

## ⚡ 性能

在 **Jetson Orin Nano 8GB** + **Astra Pro 640×480@30fps** 上实测:

### 逐帧耗时 (semantic_cloud_node)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 色彩转换 (cvt) | **~1 ms** | CUDA `gpuSwapRB` (零拷贝) |
| YOLO 推理 | **~18 ms** | TensorRT FP16 + 融合预处理 kernel |
| 点云生成 | **~6-8 ms** | 深度→3D 投影 + 语义着色 + 动态过滤 |
| 发布 | **~10-15 ms** | ROS2 序列化 + DDS |
| **总计** | **~35-42 ms** | **~24 FPS** |

### GPU 优化清单

| 优化 | 文件 | 说明 |
|---|---|---|
| 零拷贝内存 | `yolo_inference.cpp` | `cudaHostAllocMapped` 消除 CPU↔GPU 拷贝 |
| 融合预处理 Kernel | `cuda_preprocess.cu` | resize + BGR→RGB + normalize + HWC→CHW 单次 kernel |
| Identity 快速路径 | `cuda_preprocess.cu` | 640×480→640×640 scale=1.0 跳过双线性插值 |
| CUDA 色彩转换 | `cuda_colorspace.cu` | GPU RGB↔BGR / YUYV→BGR, <1ms |
| GPU 批量掩码解码 | `cuda_preprocess.cu` | 所有目标掩码并行 dot-product + sigmoid |

### 系统全局

| 模块 | 频率 |
|---|---|
| 语义点云 (semantic_cloud_node) | **~24 FPS** |
| 视觉里程计 (rgbd_odometry) | ~5–7 FPS |
| SLAM (rtabmap) | ~2 Hz |
| 语义地图发布 (semantic_map_node) | 1 Hz |
| 物体检测 (object_map_node) | 2 Hz |

---

## 🛠 依赖

| 依赖 | 版本 | 说明 |
|---|---|---|
| ROS 2 | Humble | `ros-humble-desktop` |
| RTAB-Map ROS | — | `sudo apt install ros-humble-rtabmap-ros` |
| OpenCV | 4.x | JetPack 自带 |
| TensorRT | 10.x | JetPack 自带 |
| CUDA | 12.x | JetPack 自带 |
| PCL | 1.12+ | `sudo apt install libpcl-dev` |
| 相机驱动 | — | `ros-humble-astra-camera` 或 `ros-humble-realsense2-camera` |

---

## 🔨 构建

```bash
# 1. 安装 ROS 2 依赖
sudo apt install ros-humble-rtabmap-ros ros-humble-astra-camera \
                 ros-humble-pcl-ros libpcl-dev

# 2. 克隆仓库
cd ~/Desktop
git clone <REPO_URL> ANTI
cd ANTI

# 3. 准备 YOLOv8 TensorRT Engine
#    需要在 Jetson 上导出 (参考 https://docs.ultralytics.com/modes/export/#tensorrt)
mkdir -p models
# 将 yolov8n-seg.engine 放入 models/ 目录

# 4. 编译
source /opt/ros/humble/setup.bash
colcon build --packages-select semantic_vslam

# 5. Source 工作空间
source install/setup.bash
```

> **注意**: 首次编译需要 ~90s (含 CUDA kernel 编译)。后续增量编译通常 <5s。

---

## 🚀 运行

### 完整语义 SLAM 系统

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/ANTI/install/setup.bash
ros2 launch semantic_vslam semantic_slam.launch.py \
    engine_path:=$HOME/Desktop/ANTI/models/yolov8n-seg.engine
```

### 带 RViz 可视化

```bash
ros2 launch semantic_vslam semantic_slam.launch.py \
    engine_path:=$HOME/Desktop/ANTI/models/yolov8n-seg.engine \
    rviz:=true
```

### 独立 YOLO 推理测试

```bash
# 单元测试 (对准静态帧 benchmark)
./install/semantic_vslam/lib/semantic_vslam/test_yolo_inference \
    models/yolov8n-seg.engine test_image.jpg

# 实时摄像头推理测试 (不含 SLAM)
./install/semantic_vslam/lib/semantic_vslam/test_yolo_realtime \
    models/yolov8n-seg.engine
```

---

## ⚙️ 参数配置

所有参数在 `config/params.yaml` 中定义，launch 自动加载。

### Launch 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `engine_path` | `models/yolov8n-seg.engine` | TensorRT engine 文件路径 |
| `conf_thresh` | `0.4` | YOLO 置信度阈值 |
| `launch_camera` | `true` | 是否启动 Astra Pro 相机驱动 |
| `rviz` | `false` | 是否同时启动 RViz |
| `database_path` | `/tmp/semantic_slam.db` | RTAB-Map 数据库路径 |
| `rgb_topic` | `/camera/color/image_raw` | RGB 图像 topic |
| `depth_topic` | `/camera/depth/image_raw` | 深度图 topic |
| `cam_info_topic` | `/camera/color/camera_info` | 相机内参 topic |

### semantic_cloud_node

| 参数 | 默认值 | 说明 |
|---|---|---|
| `conf_thresh` | 0.4 | YOLO 置信度阈值 (0.0–1.0) |
| `iou_thresh` | 0.45 | NMS IoU 阈值 |
| `depth_scale` | 0.001 | 深度值缩放因子 (Astra Pro: mm→m) |
| `enable_profiling` | false | 打印逐帧 `[perf]` 日志 |

### semantic_map_node

| 参数 | 默认值 | 说明 |
|---|---|---|
| `target_frame` | `map` | 累积点云的目标坐标系 |
| `voxel_size` | 0.02 | 体素滤波尺寸 (m) |
| `max_clouds` | 150 | 滑动窗口帧数 |
| `cloud_decimation` | 3 | 输入点云抽稀倍率 |
| `publish_rate` | 1.0 | 地图发布频率 (Hz) |
| `grid_cell_size` | 0.05 | 2D 栅格分辨率 (m) |
| `grid_min_height` | 0.1 | 障碍物最低高度 (m) |
| `grid_max_height` | 2.0 | 障碍物最高高度 (m) |

### object_map_node

| 参数 | 默认值 | 说明 |
|---|---|---|
| `target_frame` | `map` | 物体坐标系 |
| `min_points` | 50 | 物体最少 3D 点数 |
| `merge_distance` | 0.5 | 同类物体合并距离 (m) |
| `max_objects` | 50 | 最大跟踪物体数 |
| `publish_rate` | 2.0 | MarkerArray 发布频率 (Hz) |

---

## 📁 项目结构

```
ANTI/
├── CMakeLists.txt                          # 构建配置 (CUDA + ROS 2)
├── package.xml                             # ROS 2 包声明
├── AGENTS.md                               # 项目开发规范
├── models/
│   └── yolov8n-seg.engine                  # TensorRT engine (需自行导出)
├── config/
│   ├── params.yaml                         # 运行参数配置
│   └── semantic_slam.rviz                  # RViz 可视化预设
├── launch/
│   ├── semantic_slam.launch.py             # 完整系统 launch (6 节点)
│   └── test_rtabmap_standalone.launch.py   # 独立 RTAB-Map 测试
├── scripts/
│   └── profile_system.sh                   # 系统性能分析脚本
├── include/semantic_vslam/
│   ├── yolo_inference.hpp                  # TensorRT YOLOv8-seg 推理接口
│   ├── cuda_preprocess.hpp                 # GPU 预处理 (letterbox + normalize)
│   ├── cuda_colorspace.hpp                 # GPU 色彩空间转换
│   ├── semantic_cloud_node.hpp             # 语义点云节点
│   ├── semantic_map_node.hpp               # 语义地图累积节点
│   ├── object_map_node.hpp                 # 物体级 3D 检测节点
│   ├── semantic_colors.hpp                 # COCO 80 类语义颜色表
│   └── rtabmap_slam_node.hpp               # RTAB-Map 封装 (legacy)
└── src/
    ├── model_inference/
    │   ├── yolo_inference.cpp              # TensorRT 推理 (零拷贝 + 动态批次)
    │   ├── cuda_preprocess.cu              # 融合预处理 + 掩码解码 kernel
    │   ├── cuda_colorspace.cu              # RGB↔BGR / YUYV→BGR kernel
    │   ├── test_yolo_inference.cpp         # YOLO 静态图像测试
    │   ├── test_yolo_realtime.cpp          # YOLO 实时摄像头测试
    │   └── test_yolo_unit.cpp              # YOLO 单元测试 (benchmark)
    ├── pointcloud/
    │   ├── semantic_cloud_node.cpp         # 语义点云生成 (YOLO + depth → PointXYZRGB)
    │   ├── semantic_map_node.cpp           # 滑动窗口累积 + VoxelGrid 去重 + 2D 栅格
    │   ├── object_map_node.cpp             # 3D 欧氏聚类 + 包围盒 + 观测衰减
    │   ├── test_semantic_cloud.cpp         # 点云独立测试 (含可视化)
    │   └── test_semantic_cloud_unit.cpp    # 点云单元测试
    ├── rtabmap_bridge/
    │   └── rtabmap_slam_node.cpp           # RTAB-Map 封装 (legacy)
    └── test_pipeline.cpp                   # 端到端管线测试
```

---

## 📷 支持的相机

| 相机 | 色彩编码 | 适配方式 |
|---|---|---|
| Astra Pro | `rgb8` | CUDA `gpuSwapRB` → BGR (<1ms) |
| RealSense D435/D455 | `bgr8` | 零成本直通 |
| 其他 YUYV 相机 | `yuyv` | CUDA `gpuYUYVtoBGR` |

系统自动检测相机编码并选择最优转换路径，无需手动配置。

---

## 🔧 调试

### 性能分析

```bash
# 方法 1: 启用节点内置 profiling
# 修改 config/params.yaml 中 enable_profiling: true
# 输出示例:
# [perf] cvt=1ms yolo=18ms cloud=8ms pub=12ms total=39ms (25.6 FPS) objs=3 mask=8028/193248

# 方法 2: 运行系统级分析脚本
bash scripts/profile_system.sh
```

### 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| rtabmap 报 "TF is not set" | Astra 动态 TF 延迟 | launch 已用 static_transform_publisher 解决 |
| 旋转时墙面重影 | 地图去重精度不足 | 确保使用 `pcl::VoxelGrid` (不可替换为哈希方案) |
| 物体包围盒不显示 | 异步数据同步问题 | 已修复 (timestamp-based dedup) |
| 点云有椒盐噪声 | 深度传感器噪声 | 调小 `voxel_size` 或增大 `cloud_decimation` |

---

## 📄 License

Apache-2.0
