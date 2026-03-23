/*
 * semantic_cloud_node.cpp
 *
 * 语义点云生成 ROS2 节点实现
 *
 * 数据流:
 *   Astra Pro RGB + Depth → message_filters 时间同步 → YOLOv8-seg 推理 →
 *   逐像素分配语义标签 → pcl::PointXYZRGBL 点云 → sensor_msgs/PointCloud2 发布
 */

#include "semantic_vslam/semantic_cloud_node.hpp"
#include "semantic_vslam/cuda_colorspace.hpp"

#include <chrono>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace semantic_vslam {

// COCO 类别查找表 — bool[80] 替代 std::set (每帧调用 300K+ 次, O(1) vs O(log n))
// 动态类别: person=0, bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7,
// bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19
static bool initDynamic() {
  static bool t[80] = {};
  for (int c : {0,1,2,3,5,6,7,14,15,16,17,18,19}) t[c] = true;
  return true;
}
static bool kIsDynamic[80] = {};
static const bool kDynInit_ = (initDynamic(), [] { for (int c : {0,1,2,3,5,6,7,14,15,16,17,18,19}) kIsDynamic[c] = true; return true; }());

// 静态物体类别 (用于 3D 包围盒)
static bool kIsStaticObject[80] = {};
static const bool kStaticInit_ = ([] { for (int c : {13,56,57,58,59,60,61,62,72}) kIsStaticObject[c] = true; return true; }());

// COCO 类名 (用于 Marker 文字标签)
static const char* kCocoNames[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair dryer", "toothbrush"
};

SemanticCloudNode::SemanticCloudNode(const rclcpp::NodeOptions &options)
    : Node("semantic_cloud_node", options) {

  // ---- 声明参数 ----
  this->declare_parameter<std::string>("engine_path",
                                       "models/yolov8n-seg.engine");
  this->declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
  this->declare_parameter<std::string>("depth_topic",
                                       "/camera/depth/image_raw");
  this->declare_parameter<std::string>("cam_info_topic",
                                       "/camera/color/camera_info");
  this->declare_parameter<float>("conf_thresh", 0.4f);
  this->declare_parameter<float>("iou_thresh", 0.45f);
  this->declare_parameter<float>("depth_scale", 0.001f); // Astra Pro: mm → m
  this->declare_parameter<bool>("enable_profiling", false);
  enable_profiling_ = this->get_parameter("enable_profiling").as_bool();

  std::string engine_path = this->get_parameter("engine_path").as_string();
  std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
  std::string depth_topic = this->get_parameter("depth_topic").as_string();
  std::string cam_info_topic =
      this->get_parameter("cam_info_topic").as_string();
  conf_thresh_ =
      static_cast<float>(this->get_parameter("conf_thresh").as_double());
  iou_thresh_ =
      static_cast<float>(this->get_parameter("iou_thresh").as_double());
  depth_scale_ =
      static_cast<float>(this->get_parameter("depth_scale").as_double());

  // ---- 初始化 YOLO ----
  yolo_ = std::make_unique<YoloInference>(engine_path);
  if (!yolo_->init()) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize YOLO model from: %s",
                 engine_path.c_str());
    throw std::runtime_error("YOLO init failed");
  }
  RCLCPP_INFO(this->get_logger(), "YOLO model loaded: %s", engine_path.c_str());

  // ---- 订阅相机内参 (只需获取一次) ----
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      cam_info_topic, 1,
      std::bind(&SemanticCloudNode::cameraInfoCallback, this,
                std::placeholders::_1));

  // ---- 使用 message_filters 同步 RGB + Depth ----
  rgb_sub_.subscribe(this, rgb_topic);
  depth_sub_.subscribe(this, depth_topic);
  sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), rgb_sub_, depth_sub_);
  sync_->registerCallback(std::bind(&SemanticCloudNode::syncCallback, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2));

  // ---- 发布者 ----
  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/semantic_vslam/semantic_cloud", 5);

  // 转发 RGB/Depth 给 RTAB-Map (保持原始 header/frame_id)
  rgb_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>("/semantic_vslam/rgb", 5);
  depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/semantic_vslam/depth", 5);

  // 语义标签图发布 (CV_8UC1, 0=无语义, >0=class_id+1)
  label_map_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>("/semantic_vslam/label_map", 5);

  // ---- 物体 3D 包围盒 (Masked Back-Projection) ----
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/semantic_vslam/object_markers", 1);

  // TF2 (用于反投影到 map 坐标系)
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  RCLCPP_INFO(this->get_logger(),
              "SemanticCloudNode ready. Subscribed to: [%s] + [%s]",
              rgb_topic.c_str(), depth_topic.c_str());
}

// ---------------------------------------------------------------------------
// 相机内参回调
// ---------------------------------------------------------------------------
void SemanticCloudNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  if (has_cam_info_)
    return;

  fx_ = static_cast<float>(msg->k[0]);
  fy_ = static_cast<float>(msg->k[4]);
  cx_ = static_cast<float>(msg->k[2]);
  cy_ = static_cast<float>(msg->k[5]);
  has_cam_info_ = true;

  RCLCPP_INFO(this->get_logger(),
              "Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
              fx_, fy_, cx_, cy_);
}

// ---------------------------------------------------------------------------
// 同步回调: RGB + Depth 到齐后触发
// ---------------------------------------------------------------------------
void SemanticCloudNode::syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg) {

  if (!has_cam_info_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                         "Waiting for camera_info...");
    return;
  }

  auto t0 = std::chrono::steady_clock::now();

  // 1. 转换为 OpenCV (零拷贝共享, 无色彩转换)
  cv_bridge::CvImageConstPtr cv_rgb;
  cv_bridge::CvImageConstPtr cv_depth;
  try {
    cv_rgb = cv_bridge::toCvShare(rgb_msg);
    cv_depth = cv_bridge::toCvShare(depth_msg);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
    return;
  }

  // 判断输入是否为 RGB (Astra Pro = rgb8, RealSense = bgr8)
  bool input_is_rgb = (cv_rgb->encoding == "rgb8" || cv_rgb->encoding == "RGB8");

  // BGR 图用于点云着色, 原始图直接喂给 YOLO (is_rgb 跳过 BGR→RGB 双重交换)
  cv::Mat bgr_for_cloud;
  if (input_is_rgb) {
    cuda::gpuSwapRB(cv_rgb->image, bgr_for_cloud);  // CUDA RGB→BGR (<1ms)
  } else {
    bgr_for_cloud = cv_rgb->image;
  }

  // YOLO 需要的图像: 直接传原始数据 + is_rgb 标记
  const cv::Mat &yolo_input = cv_rgb->image;
  cv::Mat depth = cv_depth->image;

  if (yolo_input.empty() || depth.empty())
    return;

  // Astra Pro: RGB 和 depth 可能分辨率不同
  if (depth.rows != yolo_input.rows || depth.cols != yolo_input.cols) {
    cv::Mat depth_resized;
    cv::resize(depth, depth_resized, cv::Size(yolo_input.cols, yolo_input.rows),
               0, 0, cv::INTER_NEAREST);
    depth = depth_resized;
  }

  auto t1 = std::chrono::steady_clock::now();

  // 2. YOLOv8-seg 推理 (is_rgb: 跳过预处理中的 BGR→RGB 通道交换)
  std::vector<Object> objects;
  if (!yolo_->infer(yolo_input, objects, conf_thresh_, iou_thresh_, input_is_rgb)) {
    RCLCPP_ERROR(this->get_logger(), "YOLO inference failed");
    return;
  }

  auto t2 = std::chrono::steady_clock::now();

  // 3. 生成语义点云 + 标签图 (优化: 发布 PointXYZRGB, 节省带宽)
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cv::Mat label_map;
  generateSemanticCloud(bgr_for_cloud, depth, objects, cloud, label_map);

  auto t3 = std::chrono::steady_clock::now();

  // 4. 转换为 ROS2 PointCloud2 消息并发布 (只保留有效点, 去掉 NaN)
  {
    pcl::PointCloud<pcl::PointXYZRGB> dense_cloud;
    dense_cloud.reserve(cloud.size() / 2);  // 通常 50-70% 有效
    for (const auto &pt : cloud.points) {
      if (!std::isnan(pt.z)) {
        dense_cloud.push_back(pt);
      }
    }
    dense_cloud.width = dense_cloud.size();
    dense_cloud.height = 1;
    dense_cloud.is_dense = true;

    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(dense_cloud, pc2_msg);
    pc2_msg.header = rgb_msg->header;
    cloud_pub_->publish(pc2_msg);
  }

  // 5. 动态物体深度掩码: 单次扫描统计 + 记录位置, 避免双重全图遍历
  //    rtabmap 在深度=0 的区域不提特征 → 消除动态物体鬼影
  cv::Mat filtered_depth = depth;
  int masked_pixels = 0;
  int valid_pixels = 0;

  if (!label_map.empty()) {
    const bool is_16u = (depth.type() == CV_16UC1);
    // 收集动态像素坐标 (一次扫描, 无需第二次全图遍历)
    std::vector<std::pair<int,int>> dynamic_coords;
    dynamic_coords.reserve(1024);

    for (int v = 0; v < depth.rows; ++v) {
      const uint8_t *lbl_row = label_map.ptr<uint8_t>(v);
      const uint16_t *d16_row = is_16u ? depth.ptr<uint16_t>(v) : nullptr;
      const float *df_row = !is_16u ? depth.ptr<float>(v) : nullptr;
      for (int u = 0; u < depth.cols; ++u) {
        float z = is_16u ? static_cast<float>(d16_row[u]) * depth_scale_ : df_row[u];
        if (z > 0.01f && z < 10.0f) {
          valid_pixels++;
          uint8_t lbl = lbl_row[u];
          if (lbl > 0 && lbl <= 80 && kIsDynamic[lbl - 1]) {
            dynamic_coords.emplace_back(v, u);
          }
        }
      }
    }
    masked_pixels = static_cast<int>(dynamic_coords.size());

    // 安全阈值: 动态区域 <50% 才执行掩码
    if (masked_pixels > 0 && valid_pixels > 0 &&
        masked_pixels < valid_pixels / 2) {
      filtered_depth = depth.clone();
      if (is_16u) {
        for (auto [v, u] : dynamic_coords)
          filtered_depth.at<uint16_t>(v, u) = 0;
      } else {
        for (auto [v, u] : dynamic_coords)
          filtered_depth.at<float>(v, u) = 0.0f;
      }
    }
  }

  // 转发 RGB + 过滤后深度给 rtabmap
  rgb_pub_->publish(*rgb_msg);
  if (filtered_depth.data != depth.data) {
    auto filt_msg = cv_bridge::CvImage(
        depth_msg->header, cv_depth->encoding, filtered_depth).toImageMsg();
    depth_pub_->publish(*filt_msg);
  } else {
    depth_pub_->publish(*depth_msg);
  }

  // 6. 发布语义标签图 (lazy: 无订阅者时跳过序列化)
  if (label_map_pub_->get_subscription_count() > 0) {
    auto label_ros = cv_bridge::CvImage(rgb_msg->header, "mono8", label_map).toImageMsg();
    label_map_pub_->publish(*label_ros);
  }

  // 7. Masked Back-Projection: YOLO mask + depth → 3D AABB (无需 PCL 聚类)
  //    直接在当前帧使用 YOLO 实例 mask + depth 反投影计算 3D 包围盒
  Eigen::Matrix4f tf_mat = Eigen::Matrix4f::Identity();
  bool has_tf = false;
  try {
    auto tf_stamped = tf_buffer_->lookupTransform(
        target_frame_, rgb_msg->header.frame_id,
        tf2::TimePointZero, tf2::durationFromSec(0.05));
    Eigen::Isometry3d tf_eigen = tf2::transformToEigen(tf_stamped.transform);
    tf_mat = tf_eigen.matrix().cast<float>();
    has_tf = true;
  } catch (const tf2::TransformException &) {
    // TF 不可用时跳过物体检测 (不影响点云/地图)
  }
  if (has_tf) {
    computeObjectBoxes(objects, depth, tf_mat);
    publishMarkers();
  }

  if (enable_profiling_) {
    auto t4 = std::chrono::steady_clock::now();
    auto ms_cvt  = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto ms_yolo = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto ms_cloud = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto ms_pub  = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t0).count();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
        "[perf] cvt=%ldms yolo=%ldms cloud=%ldms pub=%ldms total=%ldms (%.1f FPS) objs=%zu mask=%d/%d",
        ms_cvt, ms_yolo, ms_cloud, ms_pub, ms_total,
        ms_total > 0 ? 1000.0 / ms_total : 0.0, objects.size(),
        masked_pixels, valid_pixels);
  }
}

// ---------------------------------------------------------------------------
// generateSemanticCloud
//
// 逐像素遍历 depth 图，对有效深度值的像素:
//   1) 计算 3D 坐标 (x, y, z)
//   2) 如果该像素被某个 YOLO 掩码覆盖 → 赋语义颜色 + label
//      否则 → 保留原始 RGB + label=0
// ---------------------------------------------------------------------------
void SemanticCloudNode::generateSemanticCloud(
    const cv::Mat &rgb, const cv::Mat &depth,
    const std::vector<Object> &objects,
    pcl::PointCloud<pcl::PointXYZRGB> &cloud,
    cv::Mat &out_label_map) {

  const int rows = depth.rows;
  const int cols = depth.cols;

  // 1. 语义标签图 + 置信度图
  cv::Mat label_map = cv::Mat::zeros(rows, cols, CV_8UC1);
  cv::Mat conf_map = cv::Mat::zeros(rows, cols, CV_32FC1);

  for (const auto &obj : objects) {
    const cv::Rect &r = obj.rect;
    int x0 = std::max(0, r.x);
    int y0 = std::max(0, r.y);
    int x1 = std::min(cols, r.x + r.width);
    int y1 = std::min(rows, r.y + r.height);
    if (x0 >= x1 || y0 >= y1 || obj.mask.empty()) continue;

    for (int y = y0; y < y1; ++y) {
      const uint8_t *mask_row = obj.mask.ptr<uint8_t>(y - r.y);
      uint8_t *label_row = label_map.ptr<uint8_t>(y);
      float *conf_row = conf_map.ptr<float>(y);
      for (int x = x0; x < x1; ++x) {
        int mx = x - r.x;
        if (mx >= 0 && mx < obj.mask.cols && mask_row[mx] > 0) {
          if (obj.prob > conf_row[x]) {
            label_row[x] = static_cast<uint8_t>(obj.label + 1);
            conf_row[x] = obj.prob;
          }
        }
      }
    }
  }

  // 2. 逐像素生成点云 — 优化: row-pointer 直接访问，避免 .at<>()
  cloud.clear();
  cloud.width = cols;
  cloud.height = rows;
  cloud.is_dense = false;
  cloud.points.resize(static_cast<size_t>(rows) * cols);

  const float inv_fx = 1.0f / fx_;
  const float inv_fy = 1.0f / fy_;
  const bool is_16u = (depth.type() == CV_16UC1);

  for (int v = 0; v < rows; ++v) {
    // 获取当前行指针 — 比 .at<>() 快 5-10x (无边界检查)
    const uint16_t *depth_row_16u = is_16u ? depth.ptr<uint16_t>(v) : nullptr;
    const float *depth_row_f = !is_16u ? depth.ptr<float>(v) : nullptr;
    const cv::Vec3b *rgb_row = rgb.ptr<cv::Vec3b>(v);
    const uint8_t *lbl_row = label_map.ptr<uint8_t>(v);
    pcl::PointXYZRGB *cloud_row = &cloud.points[v * cols];

    for (int u = 0; u < cols; ++u) {
      pcl::PointXYZRGB &pt = cloud_row[u];

      float z = is_16u ? static_cast<float>(depth_row_16u[u]) * depth_scale_
                       : depth_row_f[u];

      if (z <= 0.01f || z > 10.0f || std::isnan(z)) {
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
        pt.r = pt.g = pt.b = 0;
        continue;
      }

      pt.x = (static_cast<float>(u) - cx_) * z * inv_fx;
      pt.y = (static_cast<float>(v) - cy_) * z * inv_fy;
      pt.z = z;

      uint8_t lbl = lbl_row[u];
      if (lbl > 0) {
        int cls = lbl - 1;
        // 动态物体 → NaN (不进入语义地图, 避免鬼影)
        if (cls < 80 && kIsDynamic[cls]) {
          pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
          pt.r = pt.g = pt.b = 0;
          continue;
        }
        if (cls >= 0 && cls < 80) {
          pt.r = kSemanticColors[cls][0];
          pt.g = kSemanticColors[cls][1];
          pt.b = kSemanticColors[cls][2];
        } else {
          pt.r = rgb_row[u][2]; pt.g = rgb_row[u][1]; pt.b = rgb_row[u][0];
        }
      } else {
        pt.r = rgb_row[u][2]; pt.g = rgb_row[u][1]; pt.b = rgb_row[u][0];
      }
    }
  }

  out_label_map = label_map;
}

// ---------------------------------------------------------------------------
// computeObjectBoxes
//
// Masked Back-Projection: 对每个 YOLO 实例 mask 中的有效深度像素做反投影,
// 计算 3D AABB 包围盒。以此替代 PCL 欧氏聚类 (消除 KdTree + 聚类开销).
// ---------------------------------------------------------------------------
void SemanticCloudNode::computeObjectBoxes(
    const std::vector<Object> &objects,
    const cv::Mat &depth,
    const Eigen::Matrix4f &tf_mat) {

  const float inv_fx = 1.0f / fx_;
  const float inv_fy = 1.0f / fy_;
  const bool is_16u = (depth.type() == CV_16UC1);

  struct DetectedObject {
    int class_id;
    Eigen::Vector3f center;
    Eigen::Vector3f size;
  };
  std::vector<DetectedObject> detections;

  for (const auto &obj : objects) {
    if (obj.label < 0 || obj.label >= 80 || !kIsStaticObject[obj.label]) continue;

    Eigen::Vector3f min_pt( 1e9f,  1e9f,  1e9f);
    Eigen::Vector3f max_pt(-1e9f, -1e9f, -1e9f);
    int count = 0;

    // 只遍历 bbox 区域内的像素 (比全图遍历快很多)
    const int y0 = std::max(0, obj.rect.y);
    const int x0 = std::max(0, obj.rect.x);
    const int y1 = std::min(depth.rows, obj.rect.y + obj.rect.height);
    const int x1 = std::min(depth.cols, obj.rect.x + obj.rect.width);

    for (int v = y0; v < y1; ++v) {
      const int mv = v - obj.rect.y;  // mask 坐标
      if (mv < 0 || mv >= obj.mask.rows) continue;
      const uint8_t *mask_row = obj.mask.ptr<uint8_t>(mv);

      for (int u = x0; u < x1; ++u) {
        const int mu = u - obj.rect.x;
        if (mu < 0 || mu >= obj.mask.cols) continue;
        if (mask_row[mu] == 0) continue;  // 非物体像素

        // 读取深度
        float d = is_16u ? static_cast<float>(depth.at<uint16_t>(v, u)) * depth_scale_
                         : depth.at<float>(v, u);
        if (d < 0.1f || d > 5.0f) continue;

        // 反投影 → camera 3D
        float x = (static_cast<float>(u) - cx_) * d * inv_fx;
        float y = (static_cast<float>(v) - cy_) * d * inv_fy;

        // TF 变换 → map 坐标系
        Eigen::Vector4f p_cam(x, y, d, 1.0f);
        Eigen::Vector4f p_map = tf_mat * p_cam;

        min_pt = min_pt.cwiseMin(p_map.head<3>());
        max_pt = max_pt.cwiseMax(p_map.head<3>());
        count++;
      }
    }

    if (count < min_obj_points_) continue;

    Eigen::Vector3f center = (min_pt + max_pt) * 0.5f;
    Eigen::Vector3f size = max_pt - min_pt;

    // 尺寸合理性检查
    if (size.maxCoeff() < 0.05f || size.maxCoeff() > 3.0f) continue;
    detections.push_back({obj.label, center, size});
  }

  // ---- 3D AABB IoU 数据关联 ----
  auto computeIoU = [](const Eigen::Vector3f &c1, const Eigen::Vector3f &s1,
                       const Eigen::Vector3f &c2, const Eigen::Vector3f &s2) -> float {
    Eigen::Vector3f min1 = c1 - s1 * 0.5f, max1 = c1 + s1 * 0.5f;
    Eigen::Vector3f min2 = c2 - s2 * 0.5f, max2 = c2 + s2 * 0.5f;
    Eigen::Vector3f inter_min = min1.cwiseMax(min2);
    Eigen::Vector3f inter_max = max1.cwiseMin(max2);
    Eigen::Vector3f inter_size = (inter_max - inter_min).cwiseMax(0.0f);
    float inter_vol = inter_size.x() * inter_size.y() * inter_size.z();
    float vol1 = s1.x() * s1.y() * s1.z();
    float vol2 = s2.x() * s2.y() * s2.z();
    float union_vol = vol1 + vol2 - inter_vol;
    return (union_vol > 1e-6f) ? (inter_vol / union_vol) : 0.0f;
  };

  auto isContainedIn = [](const Eigen::Vector3f &c1, const Eigen::Vector3f &s1,
                          const Eigen::Vector3f &c2, const Eigen::Vector3f &s2) -> bool {
    Eigen::Vector3f min1 = c1 - s1 * 0.5f, max1 = c1 + s1 * 0.5f;
    Eigen::Vector3f min2 = c2 - s2 * 0.5f, max2 = c2 + s2 * 0.5f;
    Eigen::Vector3f inter_min = min1.cwiseMax(min2);
    Eigen::Vector3f inter_max = max1.cwiseMin(max2);
    Eigen::Vector3f inter_size = (inter_max - inter_min).cwiseMax(0.0f);
    float inter_vol = inter_size.x() * inter_size.y() * inter_size.z();
    float vol1 = s1.x() * s1.y() * s1.z();
    return (vol1 > 1e-6f) && (inter_vol / vol1 > 0.5f);
  };

  rclcpp::Time now_time = this->now();

  for (auto &det : detections) {
    int best_idx = -1;
    float best_iou = 0.0f;
    for (int i = 0; i < static_cast<int>(object_map_.size()); ++i) {
      auto &obj = object_map_[i];
      if (obj.class_id != det.class_id) continue;
      float iou = computeIoU(obj.center, obj.size, det.center, det.size);
      bool contained = isContainedIn(det.center, det.size, obj.center, obj.size) ||
                       isContainedIn(obj.center, obj.size, det.center, det.size);
      float score = contained ? std::max(iou, 0.2f) : iou;
      if (score > best_iou) {
        best_iou = score;
        best_idx = i;
      }
    }

    if (best_idx >= 0 && best_iou > 0.15f) {
      auto &obj = object_map_[best_idx];
      float w = 1.0f / (obj.observe_count + 1);
      obj.center = obj.center * (1.0f - w) + det.center * w;
      obj.size = obj.size * (1.0f - w) + det.size * w;
      obj.observe_count++;
      obj.last_seen = now_time;
    } else if (static_cast<int>(object_map_.size()) < max_objects_) {
      object_map_.push_back({det.class_id, det.center, det.size, 1, now_time});
    }
  }

  // 合并冗余物体
  for (int i = 0; i < static_cast<int>(object_map_.size()); ++i) {
    for (int j = i + 1; j < static_cast<int>(object_map_.size()); ) {
      if (object_map_[i].class_id != object_map_[j].class_id) { ++j; continue; }
      float iou = computeIoU(object_map_[i].center, object_map_[i].size,
                             object_map_[j].center, object_map_[j].size);
      bool contained = isContainedIn(object_map_[j].center, object_map_[j].size,
                                     object_map_[i].center, object_map_[i].size) ||
                       isContainedIn(object_map_[i].center, object_map_[i].size,
                                     object_map_[j].center, object_map_[j].size);
      if (iou > 0.1f || contained) {
        auto &keep = (object_map_[i].observe_count >= object_map_[j].observe_count)
                         ? object_map_[i] : object_map_[j];
        auto &drop = (object_map_[i].observe_count >= object_map_[j].observe_count)
                         ? object_map_[j] : object_map_[i];
        float w = static_cast<float>(drop.observe_count) /
                  (keep.observe_count + drop.observe_count);
        keep.center = keep.center * (1.0f - w) + drop.center * w;
        keep.size = keep.size.cwiseMax(drop.size);
        keep.observe_count += drop.observe_count;
        if (drop.last_seen > keep.last_seen) keep.last_seen = drop.last_seen;
        object_map_.erase(object_map_.begin() + j);
      } else {
        ++j;
      }
    }
  }

  // 观测计数衰减 + 清理
  for (auto &obj : object_map_) {
    if ((now_time - obj.last_seen).seconds() > 5.0 && obj.observe_count > 0) {
      obj.observe_count--;
    }
  }
  object_map_.erase(
      std::remove_if(object_map_.begin(), object_map_.end(),
          [&](const ObjectInstance &obj) {
            return (now_time - obj.last_seen).seconds() > 15.0 ||
                   obj.observe_count <= 0;
          }),
      object_map_.end());
}

// ---------------------------------------------------------------------------
// publishMarkers
// ---------------------------------------------------------------------------
void SemanticCloudNode::publishMarkers() {
  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = this->now();

  // 先清除旧 markers
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  delete_marker.header.stamp = stamp;
  delete_marker.header.frame_id = target_frame_;
  marker_array.markers.push_back(delete_marker);

  int id = 0;
  for (const auto &obj : object_map_) {
    if (obj.observe_count < 1) continue;

    // 半透明立方体
    visualization_msgs::msg::Marker cube;
    cube.header.stamp = stamp;
    cube.header.frame_id = target_frame_;
    cube.ns = "object_boxes";
    cube.id = id;
    cube.type = visualization_msgs::msg::Marker::CUBE;
    cube.action = visualization_msgs::msg::Marker::ADD;
    cube.pose.position.x = obj.center.x();
    cube.pose.position.y = obj.center.y();
    cube.pose.position.z = obj.center.z();
    cube.pose.orientation.w = 1.0;
    cube.scale.x = std::max(obj.size.x(), 0.05f);
    cube.scale.y = std::max(obj.size.y(), 0.05f);
    cube.scale.z = std::max(obj.size.z(), 0.05f);

    if (obj.class_id >= 0 && obj.class_id < 80) {
      cube.color.r = kSemanticColors[obj.class_id][0] / 255.0f;
      cube.color.g = kSemanticColors[obj.class_id][1] / 255.0f;
      cube.color.b = kSemanticColors[obj.class_id][2] / 255.0f;
    } else {
      cube.color.r = 1.0f; cube.color.g = 1.0f; cube.color.b = 0.0f;
    }
    cube.color.a = std::min(0.3f + obj.observe_count * 0.04f, 0.7f);
    cube.lifetime = rclcpp::Duration(0, 0);
    marker_array.markers.push_back(cube);

    // 文字标签
    visualization_msgs::msg::Marker text;
    text.header = cube.header;
    text.ns = "object_labels";
    text.id = id;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = obj.center.x();
    text.pose.position.y = obj.center.y();
    text.pose.position.z = obj.center.z() + obj.size.z() * 0.5f + 0.1f;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.12;
    text.color.r = 1.0f; text.color.g = 1.0f;
    text.color.b = 1.0f; text.color.a = 1.0f;

    const char *name = (obj.class_id >= 0 && obj.class_id < 80)
                           ? kCocoNames[obj.class_id] : "unknown";
    text.text = std::string(name) + " (x" +
                std::to_string(obj.observe_count) + ")";
    text.lifetime = rclcpp::Duration(0, 0);
    marker_array.markers.push_back(text);

    id++;
  }

  marker_pub_->publish(marker_array);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "Object map: %zu objects tracked", object_map_.size());
}

} // namespace semantic_vslam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_vslam::SemanticCloudNode)
