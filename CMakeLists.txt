cmake_minimum_required(VERSION 3.8)
project(semantic_vslam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17 -Wno-unused-but-set-variable -Wno-sign-compare -Wno-unused-variable -fopenmp -O3 -ftree-vectorize -funroll-loops)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# === 开发规则 3: 构建可视化 ===
set(CMAKE_VERBOSE_MAKEFILE ON)

# === 开发规则 1: 统一使用 C++ ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# === CUDA 支持 (GPU 预处理 kernel) ===
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# =============================================================================
# Dependencies
# =============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

# --- TensorRT ---
find_library(NVINFER_LIB nvinfer REQUIRED)
find_path(NVINFER_INCLUDE_DIR NvInfer.h REQUIRED)

# --- CUDA ---
find_package(CUDA REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${NVINFER_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# =============================================================================
# yolo_inference library (已完成的 YOLOv8-seg 模块)
# =============================================================================
add_library(yolo_inference SHARED
  src/model_inference/yolo_inference.cpp
  src/model_inference/cuda_preprocess.cu
  src/model_inference/cuda_colorspace.cu
)
set_target_properties(yolo_inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(yolo_inference
  ${OpenCV_LIBS}
  ${NVINFER_LIB}
  ${CUDA_CUDART_LIBRARY}
)

# === test_yolo_inference (独立 YOLO 测试) ===
add_executable(test_yolo_inference src/model_inference/test_yolo_inference.cpp)
target_link_libraries(test_yolo_inference yolo_inference ${OpenCV_LIBS})

# =============================================================================
# semantic_cloud_node (ROS2 语义点云节点)
# =============================================================================
add_executable(semantic_cloud_node src/pointcloud/semantic_cloud_node.cpp)
target_link_libraries(semantic_cloud_node
  yolo_inference
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
ament_target_dependencies(semantic_cloud_node
  rclcpp sensor_msgs cv_bridge message_filters image_transport pcl_conversions
)

# === test_semantic_cloud (独立语义点云测试，不依赖 ROS) ===
add_executable(test_semantic_cloud src/pointcloud/test_semantic_cloud.cpp)
target_link_libraries(test_semantic_cloud
  yolo_inference
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# =============================================================================
# rtabmap_slam_node (RTAB-Map SLAM 节点)
#
# 重要: RTAB-Map 系统包链接 OpenCV 4.5d，而 JetPack 提供 4.8.0。
# 为避免 ABI 不兼容 (cv::Mat setSize 崩溃)，此目标必须使用 4.5d。
# =============================================================================
find_package(RTABMap REQUIRED COMPONENTS core utilite)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# --- OpenCV 4.5d (与 RTAB-Map 一致) ---
set(OPENCV_45D_DIR "/usr/lib/aarch64-linux-gnu")
set(OPENCV_45D_INCLUDE "/usr/include/opencv4")
set(OPENCV_45D_LIBS
  ${OPENCV_45D_DIR}/libopencv_core.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_imgproc.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_imgcodecs.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_highgui.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_features2d.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_calib3d.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_flann.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_video.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_videoio.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_photo.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_aruco.so.4.5d
  ${OPENCV_45D_DIR}/libopencv_rgbd.so.4.5d
)

add_executable(rtabmap_slam_node src/rtabmap_bridge/rtabmap_slam_node.cpp)
target_include_directories(rtabmap_slam_node PRIVATE
  ${RTABMap_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OPENCV_45D_INCLUDE}
)
target_link_directories(rtabmap_slam_node PRIVATE
  ${CMAKE_SOURCE_DIR}/lib_links
)
target_link_libraries(rtabmap_slam_node
  ${OPENCV_45D_LIBS}
  ${PCL_LIBRARIES}
  rtabmap::core
  rtabmap::utilite
)
ament_target_dependencies(rtabmap_slam_node
  rclcpp sensor_msgs nav_msgs cv_bridge message_filters pcl_conversions
)

# === semantic_map_node (语义地图累积 — 用 TF 变换 + 体素滤波) ===
add_executable(semantic_map_node src/pointcloud/semantic_map_node.cpp)
target_link_libraries(semantic_map_node ${PCL_LIBRARIES})
ament_target_dependencies(semantic_map_node
  rclcpp sensor_msgs nav_msgs pcl_conversions tf2_ros tf2_geometry_msgs tf2_eigen
)

# === object_map_node (物体级语义地图 — 3D 包围盒) ===
add_executable(object_map_node src/pointcloud/object_map_node.cpp)
target_link_libraries(object_map_node ${PCL_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(object_map_node
  rclcpp sensor_msgs visualization_msgs pcl_conversions
  tf2_ros tf2_geometry_msgs tf2_eigen cv_bridge
)

# === test_yolo_unit (YOLO 单元测试) ===
add_executable(test_yolo_unit src/model_inference/test_yolo_unit.cpp)
target_link_libraries(test_yolo_unit yolo_inference ${OpenCV_LIBS})

# === test_semantic_cloud_unit (语义点云单元测试) ===
add_executable(test_semantic_cloud_unit src/pointcloud/test_semantic_cloud_unit.cpp)
target_link_libraries(test_semantic_cloud_unit
  yolo_inference
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# === test_pipeline (端到端管线测试) ===
add_executable(test_pipeline src/test_pipeline.cpp)
target_link_libraries(test_pipeline
  yolo_inference
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# === test_yolo_realtime (实时摄像头 YOLO 分割测试) ===
add_executable(test_yolo_realtime src/model_inference/test_yolo_realtime.cpp)
target_link_libraries(test_yolo_realtime yolo_inference ${OpenCV_LIBS})

# =============================================================================
# Install
# =============================================================================
install(TARGETS yolo_inference
  LIBRARY DESTINATION lib
)
install(TARGETS
  semantic_cloud_node
  rtabmap_slam_node
  semantic_map_node
  object_map_node
  test_yolo_inference
  test_semantic_cloud
  test_yolo_unit
  test_semantic_cloud_unit
  test_pipeline
  test_yolo_realtime
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_export_libraries(yolo_inference)
ament_package()

