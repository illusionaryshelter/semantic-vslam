"""
semantic_slam.launch.py

语义 SLAM 全流程 (官方 rtabmap_ros + YOLO 语义层):
  1. astra_camera          — Astra Pro RGBD 摄像头驱动
  2. static TF              — camera_link → optical frames (防止 TF 延迟)
  3. rgbd_odometry (官方)   — 10 FPS 视觉里程计
  4. rtabmap (官方)         — 2Hz SLAM + cloud_map + grid_map
  5. semantic_cloud_node    — YOLO 推理 + 语义点云 + label_map
  6. semantic_map_node      — TF 累积语义地图

相机: Astra Pro (UVC RGB + OpenNI Depth)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import AnyLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([

        # ---- Launch 参数 ----
        DeclareLaunchArgument('engine_path',
            default_value='models/yolov8n-seg.engine',
            description='Path to TensorRT engine file'),
        DeclareLaunchArgument('conf_thresh',
            default_value='0.4',
            description='YOLO confidence threshold'),
        DeclareLaunchArgument('launch_camera',
            default_value='true',
            description='Whether to launch Astra Pro camera driver'),
        DeclareLaunchArgument('rviz',
            default_value='false',
            description='Launch RViz2 with rtabmap config'),
        DeclareLaunchArgument('database_path',
            default_value='/tmp/semantic_slam.db',
            description='RTAB-Map database path'),

        # Astra Pro topic 名称
        DeclareLaunchArgument('rgb_topic',
            default_value='/camera/color/image_raw',
            description='RGB image topic'),
        DeclareLaunchArgument('depth_topic',
            default_value='/camera/depth/image_raw',
            description='Depth image topic'),
        DeclareLaunchArgument('cam_info_topic',
            default_value='/camera/color/camera_info',
            description='Camera info topic'),

        # ============================================================
        # 1. Astra Pro 摄像头驱动
        # ============================================================
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('astra_camera'),
                    'launch',
                    'astra_pro.launch.xml'
                ])
            ),
            launch_arguments={
                'camera_name': 'camera',
                'depth_registration': 'true',
                'color_depth_synchronization': 'true',
                'color_width': '640',
                'color_height': '480',
                'color_fps': '30',
                'depth_width': '640',
                'depth_height': '480',
                'depth_fps': '30',
                'enable_point_cloud': 'false',
                'enable_colored_point_cloud': 'false',
            }.items(),
            condition=IfCondition(LaunchConfiguration('launch_camera')),
        ),

        # ============================================================
        # 2. 静态 TF: camera_link → camera_color_optical_frame
        #
        # Astra 驱动以 10Hz 动态发布此 TF, 但在 Jetson CPU 负载高时
        # (YOLO + rtabmap + odometry 同时运行) TF 会严重延迟 (>2s)
        # 导致 rtabmap 报错 "TF is not set"。
        # 用 static TF 保证此变换始终可用。
        #
        # 标准 optical frame 转换:
        #   camera_link (x=forward, y=left, z=up) →
        #   optical_frame (x=right, y=down, z=forward)
        #   旋转: roll=-pi/2, pitch=0, yaw=-pi/2
        # ============================================================
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_color_optical_tf',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '-1.5707963', '--pitch', '0', '--yaw', '-1.5707963',
                '--frame-id', 'camera_link',
                '--child-frame-id', 'camera_color_optical_frame',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_depth_optical_tf',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '-1.5707963', '--pitch', '0', '--yaw', '-1.5707963',
                '--frame-id', 'camera_link',
                '--child-frame-id', 'camera_depth_optical_frame',
            ],
        ),

        # ============================================================
        # 3. 官方 RTAB-Map (里程计 + SLAM + 密集建图 + Grid Map)
        # ============================================================
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('rtabmap_launch'),
                    'launch',
                    'rtabmap.launch.py'
                ])
            ),
            launch_arguments={
                # ---- Topic 映射 (Astra Pro) ----
                'rgb_topic':            LaunchConfiguration('rgb_topic'),
                'depth_topic':          LaunchConfiguration('depth_topic'),
                'camera_info_topic':    LaunchConfiguration('cam_info_topic'),

                # ---- 坐标系 ----
                'frame_id':             'camera_link',

                # ---- 数据库 ----
                'database_path':        LaunchConfiguration('database_path'),

                # ---- 可视化 ----
                'rviz':                 LaunchConfiguration('rviz'),
                'rtabmap_viz':          'false',

                # ---- 同步 (关键修复) ----
                'approx_sync':          'true',
                'approx_sync_max_interval': '0.05',  # 允许 50ms RGB-Depth 时间差
                'subscribe_depth':      'true',
                'queue_size':           '20',
                'qos':                  '1',   # best-effort

                # ---- TF (关键修复) ----
                'wait_for_transform':   '1.0',  # 等待 TF 1 秒 (默认 0.2s 不够)

                # ---- 里程计 ----
                'visual_odometry':      'true',
                'odom_frame_id':        'odom',
                'publish_tf':           'true',

                # ---- RTAB-Map 参数 ----
                'args': '--delete_db_on_start '
                        '--Rtabmap/DetectionRate 2 '
                        '--RGBD/OptimizeMaxError 3.0 '
                        '--Mem/RehearsalSimilarity 0.6 '
                        # 禁用 rtabmap 自带地图 (我们有语义版本)
                        '--Grid/FromDepth false '   # 不从深度图生成栅格 → 节省大量 CPU
                        '--Grid/3D false '           # 不生成 3D cloud_map
                        '--Grid/RangeMax 0 ',        # 禁用栅格射线追踪
            }.items(),
        ),

        # ============================================================
        # 4. 语义处理组件容器 (IPC 进程内零拷贝通信)
        #
        # 将 3 个自定义节点加载到同一进程:
        #   - semantic_cloud_node (YOLO 推理 + 语义点云)
        #   - semantic_map_node   (TF 累积 + 体素滤波)
        #   - object_map_node     (3D 聚类 + 包围盒)
        #
        # use_intra_process_comms=True:
        #   节点间传递 unique_ptr 消息, 零序列化/零拷贝
        #   ~19MB/帧 DDS 序列化开销 → 0
        # ============================================================
        ComposableNodeContainer(
            name='semantic_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',  # 多线程: 各节点回调并行执行
            output='screen',
        ),

        # ---- 加载: semantic_cloud_node ----
        LoadComposableNodes(
            target_container='semantic_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='semantic_vslam',
                    plugin='semantic_vslam::SemanticCloudNode',
                    name='semantic_cloud_node',
                    parameters=[
                        PathJoinSubstitution([
                            FindPackageShare('semantic_vslam'),
                            'config', 'params.yaml'
                        ]),
                        {
                        'engine_path':    LaunchConfiguration('engine_path'),
                        'rgb_topic':      LaunchConfiguration('rgb_topic'),
                        'depth_topic':    LaunchConfiguration('depth_topic'),
                        'cam_info_topic': LaunchConfiguration('cam_info_topic'),
                        'conf_thresh':    LaunchConfiguration('conf_thresh'),
                        'depth_scale':    0.001,
                        },
                    ],
                    extra_arguments=[
                        {'use_intra_process_comms': True},
                    ],
                ),
            ],
        ),

        # ---- 加载: semantic_map_node ----
        LoadComposableNodes(
            target_container='semantic_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='semantic_vslam',
                    plugin='semantic_vslam::SemanticMapNode',
                    name='semantic_map_node',
                    parameters=[
                        PathJoinSubstitution([
                            FindPackageShare('semantic_vslam'),
                            'config', 'params.yaml'
                        ]),
                        {
                        'target_frame':     'map',
                        'voxel_size':       0.02,
                        'max_clouds':       150,
                        'cloud_decimation': 3,
                        'publish_rate':     1.0,
                        },
                    ],
                    extra_arguments=[
                        {'use_intra_process_comms': True},
                    ],
                ),
            ],
        ),

        # ---- 加载: object_map_node ----
        LoadComposableNodes(
            target_container='semantic_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='semantic_vslam',
                    plugin='semantic_vslam::ObjectMapNode',
                    name='object_map_node',
                    parameters=[
                        PathJoinSubstitution([
                            FindPackageShare('semantic_vslam'),
                            'config', 'params.yaml'
                        ]),
                    ],
                    extra_arguments=[
                        {'use_intra_process_comms': True},
                    ],
                ),
            ],
        ),
    ])
