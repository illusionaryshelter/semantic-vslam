# Semantic VSLAM

> **基于 YOLOv8-seg + RTAB-Map 的实时语义视觉 SLAM 系统**
>
> 目标硬件: NVIDIA Jetson Orin Nano 8GB | C++ / CUDA | ROS 2 Humble
>
> ⚠️ **非 Jetson版本请切换到 [`discrete-gpu`](../../tree/discrete-gpu) 分支**

将 YOLOv8 实例分割与 RTAB-Map 视觉 SLAM 融合，在 RGB-D 输入上实时生成**语义着色 3D 点云地图**、**2D 占据栅格地图**和**物体级 3D 包围盒**，为下游导航和场景理解提供语义感知能力。

---

##  系统架构

```
                          ┌──────────────────────────────────────────────┐
                          │          semantic_cloud_node                 │
 ┌─────────────┐          │  ┌──────────┐  ┌─────────────────────────┐   │
 │ Astra Pro / │  RGB     │  │ TensorRT │  │ generateSemanticCloud   │   │
 │ RealSense   │─────────▶│  │ YOLOv8   │─▶│ depth→3D + 语义着色     │  │──▶ /semantic_cloud
 │ (RGB-D)     │  Depth   │  │ -seg     │  │ + 动态物体过滤            │  │   /label_map
 └──────┬──────┘─────────▶│  └──────────┘  └─────────────────────────┘  │
        │                 └──────────────────────────────────────────────┘
        │                                          │
        │            ┌──────────────────┐          │          ┌──────────────────┐
        │            │ rgbd_odometry    │          ├─────────▶│ semantic_map_node│
        └───────────▶│ (官方 rtabmap)   │          │          │ 增量式 CUDA      │
                     └────────┬─────────┘          │          │ VoxelGrid 全局地图│
                              │                    │          └────────┬─────────┘
                     ┌────────▼─────────┐          │          /semantic_map_cloud (3D 语义地图)
                     │ rtabmap SLAM     │          │          /grid_map           (2D 栅格地图)
                     │ (官方 rtabmap)    │          │
                     └──────────────────┘          │          ┌──────────────────┐
                                                   └─────────▶│ object_map_node  │
                                                              │ 3D AABB + 跟踪   │
                                                              └────────┬─────────┘
                                                                       │
                                                              /object_markers (MarkerArray)
```

### 输出 Topics

| Topic | 类型 | 说明 |
|---|---|---|
| `/semantic_vslam/semantic_cloud` | PointCloud2 | 单帧语义着色 3D 点云 |
| `/semantic_vslam/semantic_map_cloud` | PointCloud2 | 增量式全局语义地图 (永久保留) |
| `/semantic_vslam/grid_map` | OccupancyGrid | 语义 2D 占据栅格 |
| `/semantic_vslam/label_map` | Image (CV_8UC1) | 逐像素语义标签图 |
| `/semantic_vslam/object_markers` | MarkerArray | 物体级 3D 包围盒 |

---

##  性能

在 **Jetson Orin Nano 8GB** + **Astra Pro 640×480@30fps** 上实测:

### 逐帧耗时 (semantic_cloud_node)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 色彩转换 (cvt) | **~1 ms** | CUDA `gpuSwapRB` (零拷贝) |
| YOLO 推理 | **~12-19 ms** | TensorRT FP16 + 融合预处理 kernel |
| 点云生成 | **~6-8 ms** | 深度→3D 投影 + 语义着色 + 动态过滤 |
| 发布 | **~10-15 ms** | ROS2 序列化 + DDS |
| **总计** | **~29-40 ms** | **~25-34 FPS** |

### 地图累积耗时 (semantic_map_node)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| CUDA VoxelGrid | **~25-35 ms** | 增量式: global_map + new_frame → voxelize |
| 2D 栅格投影 | **~1 ms** | 3D → OccupancyGrid |
| **总计** | **~30-40 ms** | 增量式无需 merge 开销 |

### GPU 优化清单

| 优化 | 文件 | 说明 |
|---|---|---|
| 零拷贝内存 | `yolo_inference.cpp` | `cudaHostAllocMapped` 消除 CPU↔GPU 拷贝 |
| 融合预处理 Kernel | `cuda_preprocess.cu` | resize + BGR→RGB + normalize + HWC→CHW 单次 kernel |
| Identity 快速路径 | `cuda_preprocess.cu` | 640×480→640×640 scale=1.0 跳过双线性插值 |
| CUDA 色彩转换 | `cuda_colorspace.cu` | GPU RGB↔BGR / YUYV→BGR, <1ms |
| GPU 批量掩码解码 | `cuda_preprocess.cu` | 所有目标掩码并行 dot-product + sigmoid |
| **CUDA VoxelGrid** | `cuda_voxel_grid.cu` | `thrust::sort_by_key` 替代 PCL VoxelGrid |
| **UMA 零拷贝** | `cuda_voxel_grid.cu` | `cudaMallocManaged` 消除 H2D/D2H (Jetson UMA) |
| **持久显存池** | `cuda_voxel_grid.cu` | `GPUPool` 一次分配跨调用复用, 避免反复 cudaMalloc/Free |
| **uint64 空间哈希** | `cuda_voxel_grid.cu` | 21-bit/轴 bit-packing, 无碰撞, ±20km 范围 |
| **增量式体素化** | `cuda_voxel_grid_wrapper.cpp` | `CudaIncrementalVoxelGrid` 永久全局地图 |

### 系统全局

| 模块 | 频率 |
|---|---|
| 语义点云 (semantic_cloud_node) | **~25-34 FPS** |
| 视觉里程计 (rgbd_odometry) | ~5–7 FPS |
| SLAM (rtabmap) | ~2 Hz |
| 语义地图发布 (semantic_map_node) | 1 Hz |
| 物体检测 (object_map_node) | 2 Hz |

---

##  依赖

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

## 运行

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

### CUDA VoxelGrid 测试

```bash
# 单元测试: 500K 点 @ 0.02m 体素
./install/semantic_vslam/lib/semantic_vslam/test_cuda_voxel_grid

# cuPCL 对标 benchmark (需要 sample.pcd)
./install/semantic_vslam/lib/semantic_vslam/test_cupcl_benchmark sample.pcd 1.0
```

---

##  参数配置

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
| `cloud_decimation` | 2 | 输入点云抽稀倍率 |
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

##  项目结构

```
ANTI/
├── CMakeLists.txt                          # 构建配置 (CUDA + ROS 2)
├── package.xml                             # ROS 2 包声明
├── .clangd                                 # clangd 配置 (CUDA 12.6 兼容)
├── models/
│   └── yolov8n-seg.engine                  # TensorRT engine (需自行导出)
├── config/
│   ├── params.yaml                         # 运行参数配置
│   └── semantic_slam.rviz                  # RViz 可视化预设
├── launch/
│   ├── semantic_slam.launch.py             # 完整系统 launch
│   └── test_rtabmap_standalone.launch.py   # 独立 RTAB-Map 测试
├── scripts/
│   └── profile_system.sh                   # 系统性能分析脚本
├── include/semantic_vslam/
│   ├── yolo_inference.hpp                  # TensorRT YOLOv8-seg 推理接口
│   ├── cuda_preprocess.hpp                 # GPU 预处理 (letterbox + normalize)
│   ├── cuda_colorspace.hpp                 # GPU 色彩空间转换
│   ├── cuda_voxel_grid.hpp                 # CUDA VoxelGrid + 增量式全局地图
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
    │   ├── semantic_map_node.cpp           # 增量式全局地图 + 2D 栅格
    │   ├── object_map_node.cpp             # 3D 包围盒 + 物体跟踪
    │   ├── cuda_voxel_grid.cu              # CUDA VoxelGrid kernel (Thrust)
    │   ├── cuda_voxel_grid_wrapper.cpp     # PCL ↔ VoxelPoint 封装 + 增量式类
    │   ├── test_cuda_voxel_grid.cpp        # CUDA VoxelGrid 单元测试
    │   ├── test_cupcl_benchmark.cpp        # cuPCL 对标 benchmark
    │   ├── test_semantic_cloud.cpp         # 点云独立测试 (含可视化)
    │   └── test_semantic_cloud_unit.cpp    # 点云单元测试
    ├── rtabmap_bridge/
    │   └── rtabmap_slam_node.cpp           # RTAB-Map 封装 (legacy)
    └── test_pipeline.cpp                   # 端到端管线测试
```

---

##  CUDA VoxelGrid 技术细节

### 算法

```
输入点云 → calcVoxelKey (uint64 bit-packing) → thrust::sort_by_key
         → markBoundary → inclusive_scan → gatherFirstPoint → 输出
```

### 空间哈希 (uint64_t bit-packing)

每轴 21 位编码有符号体素坐标, 打包为 64 位无碰撞 key:

```
key = (vx + OFFSET) << 42 | (vy + OFFSET) << 21 | (vz + OFFSET)
```

- 范围: ±1,048,575 个 voxel = **±20km** @ 0.02m 体素
- 无哈希碰撞 (排序后相同 voxel 的点必然相邻)
- 对比 Nießner XOR Hash: `(x*73856093) ⊕ (y*19349669) ⊕ (z*83492791)` 有碰撞, 不适用于 sort-based 算法

### Jetson UMA 优化

| 优化 | 效果 |
|---|---|
| `cudaMallocManaged` | CPU/GPU 共享物理内存, 消除 H2D/D2H 拷贝 |
| 持久 `GPUPool` | 一次 alloc, 跨调用复用 (1.5x 增长策略) |
| 直写输出 | gather kernel 直接写到调用者缓冲区, 无中间 memcpy |

### 增量式全局地图

```
旧方案 (滑动窗口):
  150帧 → merge(120-234ms) → voxelGrid(200-290ms) → 地图会消失!

新方案 (增量式):
  global_map(50K) + new_frame(5K) → voxelGrid(~25-35ms) → 地图永久保留
```

---

##  支持的相机

| 相机 | 色彩编码 | 适配方式 |
|---|---|---|
| Astra Pro | `rgb8` | CUDA `gpuSwapRB` → BGR (<1ms) |
| RealSense D435/D455 | `bgr8` | 零成本直通 |
| 其他 YUYV 相机 | `yuyv` | CUDA `gpuYUYVtoBGR` |

系统自动检测相机编码并选择最优转换路径，无需手动配置。

---

##  调试

### 性能分析

```bash
# 方法 1: 启用节点内置 profiling
# 修改 config/params.yaml 中 enable_profiling: true
# 输出示例:
# [perf] cvt=1ms yolo=18ms cloud=8ms pub=12ms total=39ms (25.6 FPS) objs=3 mask=8028/193248
# [perf] map: voxel=28ms pub=1ms total=30ms | 52K pts (incremental)

# 方法 2: 运行系统级分析脚本
bash scripts/profile_system.sh
```

---

## 📄 License

Apache-2.0
