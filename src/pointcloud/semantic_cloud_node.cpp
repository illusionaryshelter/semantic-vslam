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
#include <set>
#include <pcl_conversions/pcl_conversions.h>

namespace semantic_vslam {

// COCO 动态类别 ID (可能在场景中移动的物体)
// person=0, bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7,
// bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19
static const std::set<int> kDynamicClasses = {
    0, 1, 2, 3, 5, 6, 7, 14, 15, 16, 17, 18, 19
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

  // 4. 转换为 ROS2 PointCloud2 消息并发布
  sensor_msgs::msg::PointCloud2 pc2_msg;
  pcl::toROSMsg(cloud, pc2_msg);
  pc2_msg.header = rgb_msg->header;
  cloud_pub_->publish(pc2_msg);

  // 5. 动态物体深度掩码: 将人/车/动物等动态类别区域的深度置零
  //    rtabmap 在深度=0 的区域不提特征 → 消除动态物体鬼影
  //    安全机制: 如果剔除比例 >50%, 跳过掩码防止特征过少跟丢
  cv::Mat filtered_depth = depth;  // 默认: 不拷贝, 直接用原始深度
  int masked_pixels = 0;
  int valid_pixels = 0;

  if (!label_map.empty()) {
    // 先统计: 动态区域占多少有效深度像素?
    const bool is_16u = (depth.type() == CV_16UC1);
    for (int v = 0; v < depth.rows; ++v) {
      const uint8_t *lbl_row = label_map.ptr<uint8_t>(v);
      const uint16_t *d16_row = is_16u ? depth.ptr<uint16_t>(v) : nullptr;
      const float *df_row = !is_16u ? depth.ptr<float>(v) : nullptr;
      for (int u = 0; u < depth.cols; ++u) {
        float z = is_16u ? static_cast<float>(d16_row[u]) * depth_scale_ : df_row[u];
        if (z > 0.01f && z < 10.0f) {
          valid_pixels++;
          uint8_t lbl = lbl_row[u];
          if (lbl > 0 && kDynamicClasses.count(lbl - 1)) {
            masked_pixels++;
          }
        }
      }
    }

    // 安全阈值: 动态区域占有效深度 <50% 才执行掩码
    if (masked_pixels > 0 && valid_pixels > 0 &&
        masked_pixels < valid_pixels / 2) {
      filtered_depth = depth.clone();
      for (int v = 0; v < filtered_depth.rows; ++v) {
        const uint8_t *lbl_row = label_map.ptr<uint8_t>(v);
        if (is_16u) {
          uint16_t *d_row = filtered_depth.ptr<uint16_t>(v);
          for (int u = 0; u < filtered_depth.cols; ++u) {
            if (lbl_row[u] > 0 && kDynamicClasses.count(lbl_row[u] - 1))
              d_row[u] = 0;
          }
        } else {
          float *d_row = filtered_depth.ptr<float>(v);
          for (int u = 0; u < filtered_depth.cols; ++u) {
            if (lbl_row[u] > 0 && kDynamicClasses.count(lbl_row[u] - 1))
              d_row[u] = 0.0f;
          }
        }
      }
    }
  }

  // 转发 RGB + 过滤后深度给 rtabmap
  rgb_pub_->publish(*rgb_msg);
  if (filtered_depth.data != depth.data) {
    // 深度被过滤过 → 重新封装为 ROS 消息
    auto filt_msg = cv_bridge::CvImage(
        depth_msg->header, cv_depth->encoding, filtered_depth).toImageMsg();
    depth_pub_->publish(*filt_msg);
  } else {
    depth_pub_->publish(*depth_msg);
  }

  // 6. 发布语义标签图
  auto label_msg = cv_bridge::CvImage(rgb_msg->header, "mono8", label_map).toImageMsg();
  label_map_pub_->publish(*label_msg);

  auto t4 = std::chrono::steady_clock::now();

  // 性能统计 (每 3 秒输出一次)
  auto ms_cvt  = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  auto ms_yolo = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  auto ms_cloud = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  auto ms_pub  = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
  auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t0).count();
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
      "Frame: cvt=%ldms yolo=%ldms cloud=%ldms pub=%ldms TOTAL=%ldms (%.1f FPS) objs=%zu mask=%d/%d",
      ms_cvt, ms_yolo, ms_cloud, ms_pub, ms_total,
      ms_total > 0 ? 1000.0 / ms_total : 0.0, objects.size(),
      masked_pixels, valid_pixels);
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

} // namespace semantic_vslam

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<semantic_vslam::SemanticCloudNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
