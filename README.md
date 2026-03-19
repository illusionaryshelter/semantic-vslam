# Semantic VSLAM

> **基于 YOLOv8-seg + RTAB-Map 的实时语义视觉 SLAM 系统**
>
> 目标硬件: NVIDIA Jetson Orin Nano 8GB | 语言: C++ / CUDA | 框架: ROS 2 Humble

将 YOLOv8 实例分割与 RTAB-Map 视觉 SLAM 融合，在 RGB-D 相机输入上实时生成**语义着色的 3D 点云地图**和 **2D 占据栅格地图**，为下游导航和场景理解提供语义感知能力。

---

##  系统架构

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Astra Pro / │     │ semantic_cloud   │     │  semantic_map    │
│ RealSense   │───▶│ _node            │───▶│  _node           │
│ (RGB-D)     │     │ YOLO推理+语义着色 │     │ TF累积+体素滤波   │
└──────┬──────┘     └──────────────────┘     └────────┬─────────┘
       │                                              │
       │            ┌──────────────────┐              ▼
       │            │ rgbd_odometry    │     /semantic_vslam/
       └──────────▶│ (官方 rtabmap)    │     semantic_map_cloud  (3D 语义地图)
                    └────────┬─────────┘     grid_map             (2D 栅格地图)
                             │
                    ┌────────▼─────────┐
                    │ rtabmap SLAM     │
                    │ (官方 rtabmap)    │────▶ /rtabmap/cloud_map  (3D RGB 地图)
                    └──────────────────┘      /rtabmap/map        (2D 栅格地图)
```

**输出 4 套地图**:

| Topic | 类型 | 说明 |
|---|---|---|
| `/semantic_vslam/semantic_map_cloud` | PointCloud2 | 语义着色 3D 点云 (YOLO 类别颜色) |
| `/semantic_vslam/grid_map` | OccupancyGrid | 语义 2D 占据栅格 |
| `/rtabmap/cloud_map` | PointCloud2 | 官方 RTAB-Map 3D 地图 (原始 RGB) |
| `/rtabmap/map` | OccupancyGrid | 官方 RTAB-Map 2D 栅格 (导航用) |

---

## 性能优化与实测

在 **Jetson Orin Nano 8GB** + **Astra Pro 640×480@30fps** 上实测:

### 逐帧耗时分解

| 阶段 | 优化前 | 优化后 | 手段 |
|---|---|---|---|
| 色彩转换 (cvt) | 38 ms | **1 ms** | CUDA `gpuSwapRB` 替代 cv_bridge CPU 转换 |
| YOLO 推理 | 17 ms | **17 ms** | TensorRT FP16, 零拷贝内存 |
| 点云生成 | 19 ms | **8 ms** | row-pointer 替代 `.at<>()`, PointXYZRGB 替代 XYZRGBL |
| 发布 | 16 ms | **11 ms** | — |
| **总计** | **91 ms (11 FPS)** | **39 ms (25.6 FPS)** | **2.3× 提升** |

### GPU 优化清单

| 优化 | 文件 | 说明 |
|---|---|---|
| **零拷贝内存** | `yolo_inference.cpp` | `cudaHostAllocMapped` 消除 CPU↔GPU 拷贝 |
| **融合预处理 Kernel** | `cuda_preprocess.cu` | resize + BGR→RGB + normalize + HWC→CHW 单 kernel |
| **Identity Letterbox 快速路径** | `cuda_preprocess.cu` | 640×480→640×640 scale=1.0 跳过双线性插值 |
| **`is_rgb` 通道交换跳过** | `cuda_preprocess.cu` | RGB 输入跳过 BGR→RGB, 消除双重交换 |
| **CUDA 色彩空间转换** | `cuda_colorspace.cu` | GPU RGB↔BGR 通道交换, <1ms |
| **GPU 批量掩码解码** | `cuda_preprocess.cu` | 所有目标掩码并行 dot-product + sigmoid |

### 系统级性能

| 模块 | 帧率 |
|---|---|
| 视觉里程计 (rgbd_odometry) | ~5–7 FPS, quality 300–400 |
| SLAM (rtabmap) | 2 Hz, 130+ nodes/min |
| 语义点云 (semantic_cloud_node) | **25.6 FPS** |
| 语义地图发布 (semantic_map_node) | 1 Hz (可配置) |

---

## 依赖

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

##  构建

```bash
# 1. 安装 ROS 2 依赖
sudo apt install ros-humble-rtabmap-ros ros-humble-astra-camera \
                 ros-humble-pcl-ros libpcl-dev

# 2. 克隆仓库
cd ~/Desktop
git clone <REPO_URL> ANTI
cd ANTI

# 3. 准备 YOLOv8 TensorRT Engine
#    (需要先在 Jetson 上导出 .engine 文件)
#    参考: https://docs.ultralytics.com/modes/export/#tensorrt
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

##  运行

### 完整语义 SLAM 系统

```bash
# 终端 1: 启动系统
source /opt/ros/humble/setup.bash
source ~/Desktop/ANTI/install/setup.bash
ros2 launch semantic_vslam semantic_slam.launch.py \
    engine_path:=$HOME/Desktop/ANTI/models/yolov8n-seg.engine

# 终端 2: 启动 RViz 可视化
rviz2 -d $(ros2 pkg prefix semantic_vslam)/share/semantic_vslam/config/semantic_slam.rviz
```

### 独立 RTAB-Map 测试 (不含 YOLO)

```bash
ros2 launch semantic_vslam test_rtabmap_standalone.launch.py
```

### 独立 YOLO 推理测试

```bash
# 单元测试 (50 FPS benchmark)
./install/semantic_vslam/lib/semantic_vslam/test_yolo_inference \
    models/yolov8n-seg.engine test_image.jpg
```

---

##  参数配置

### Launch 参数

```bash
ros2 launch semantic_vslam semantic_slam.launch.py \
    engine_path:=<path>               # TensorRT engine 文件路径
    conf_thresh:=0.4                   # YOLO 置信度阈值 (0.0-1.0)
    launch_camera:=true                # 是否启动相机驱动
    rviz:=false                        # 是否同时启动 RViz
    database_path:=/tmp/semantic_slam.db  # RTAB-Map 数据库路径
    rgb_topic:=/camera/color/image_raw    # RGB 图像 topic
    depth_topic:=/camera/depth/image_raw  # 深度图 topic
    cam_info_topic:=/camera/color/camera_info  # 相机内参 topic
```

### semantic_cloud_node 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `engine_path` | — | TensorRT engine 文件路径 |
| `conf_thresh` | 0.4 | YOLO 置信度阈值 |
| `iou_thresh` | 0.45 | NMS IoU 阈值 |
| `depth_scale` | 0.001 | 深度值缩放 (Astra Pro: 0.001 = mm→m) |

### semantic_map_node 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `target_frame` | `map` | 累积点云的目标坐标系 |
| `voxel_size` | 0.02 | 体素滤波尺寸 (m), 越小越精细 |
| `max_clouds` | 150 | 滑动窗口帧数 |
| `cloud_decimation` | 3 | 输入点云抽稀倍率 |
| `publish_rate` | 1.0 | 地图发布频率 (Hz) |
| `grid_cell_size` | 0.05 | 2D 栅格分辨率 (m) |
| `grid_min_height` | 0.1 | 障碍物最低高度 (m) |
| `grid_max_height` | 2.0 | 障碍物最高高度 (m) |

---

## 项目结构

```
ANTI/
├── CMakeLists.txt                          # 构建配置 (CUDA + ROS 2)
├── package.xml                             # ROS 2 包声明
├── models/
│   └── yolov8n-seg.engine                  # TensorRT engine (需自行导出)
├── config/
│   └── semantic_slam.rviz                  # RViz 可视化配置
├── launch/
│   ├── semantic_slam.launch.py             # 完整系统 launch
│   └── test_rtabmap_standalone.launch.py   # 独立 RTAB-Map 测试
├── include/semantic_vslam/
│   ├── yolo_inference.hpp                  # YOLO 推理接口
│   ├── cuda_preprocess.hpp                 # GPU 预处理接口
│   ├── cuda_colorspace.hpp                 # GPU 色彩空间转换接口
│   ├── semantic_cloud_node.hpp             # 语义点云节点
│   ├── semantic_map_node.hpp               # 语义地图累积节点
│   ├── semantic_colors.hpp                 # COCO 80 类语义颜色表
│   └── rtabmap_slam_node.hpp               # RTAB-Map 封装 (legacy)
└── src/
    ├── model_inference/
    │   ├── yolo_inference.cpp              # TensorRT 推理 (零拷贝)
    │   ├── cuda_preprocess.cu              # 融合预处理 kernel
    │   ├── cuda_colorspace.cu              # RGB↔BGR / YUYV→BGR kernel
    │   ├── test_yolo_inference.cpp         # YOLO 独立测试
    │   └── test_yolo_unit.cpp              # YOLO 单元测试 (benchmark)
    ├── pointcloud/
    │   ├── semantic_cloud_node.cpp         # 语义点云生成 (ROS 2 节点)
    │   ├── semantic_map_node.cpp           # 语义地图累积 (TF + voxel)
    │   └── test_semantic_cloud.cpp         # 点云独立测试
    └── rtabmap_bridge/
        └── rtabmap_slam_node.cpp           # RTAB-Map 封装 (legacy)
```

---

##  支持的相机

| 相机 | 编码 | 适配方式 |
|---|---|---|
| Astra Pro | `rgb8` | CUDA `gpuSwapRB` → BGR (<1ms) |
| RealSense D435/D455 | `bgr8` | 零成本直通 (无需转换) |
| 其他 YUYV 相机 | `yuyv` | CUDA `gpuYUYVtoBGR` |

系统自动检测相机编码并选择最优转换路径，无需手动配置。

---

## 📄 License

Apache-2.0
