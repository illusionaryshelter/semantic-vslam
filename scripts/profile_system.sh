#!/bin/bash
# =============================================================
# profile_system.sh
#
# 语义 SLAM 系统性能 Profiling 脚本
#
# 用法:
#   1. 先启动系统: ros2 launch semantic_vslam semantic_slam.launch.py ...
#   2. 另开终端:  bash scripts/profile_system.sh [duration_seconds]
#
# 输出:
#   - Topic 发布频率
#   - 节点 CPU / 内存占用
#   - 汇总报告
# =============================================================

DURATION=${1:-15}
echo "=========================================="
echo "  Semantic SLAM Profiler"
echo "  Duration: ${DURATION}s"
echo "=========================================="

source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null

echo ""
echo "--- [1/4] Topic Rates (测量 ${DURATION}s) ---"
echo ""

# 并行测量关键 topic 频率
declare -A TOPICS=(
  ["rgb_raw"]="/camera/color/image_raw"
  ["depth_raw"]="/camera/depth/image_raw"
  ["semantic_cloud"]="/semantic_vslam/semantic_cloud"
  ["label_map"]="/semantic_vslam/label_map"
  ["map_cloud"]="/semantic_vslam/semantic_map_cloud"
  ["grid_map"]="/semantic_vslam/grid_map"
  ["grid_visual"]="/semantic_vslam/grid_map_visual"
  ["object_markers"]="/semantic_vslam/object_markers"
  ["rtabmap_odom"]="/odom"
)

TMPDIR=$(mktemp -d)
for key in "${!TOPICS[@]}"; do
  topic="${TOPICS[$key]}"
  (timeout ${DURATION} ros2 topic hz "$topic" --window 50 2>/dev/null | tail -1 > "$TMPDIR/$key.txt") &
done

# 同时采集 CPU/内存
echo "--- [2/4] CPU/Memory (采样中...) ---"
echo ""

# 找到相关进程
sleep 2
PIDS=$(ps aux | grep -E 'semantic_cloud|semantic_map|object_map|rtabmap|rgbd_odometry' | grep -v grep | awk '{print $2}' | tr '\n' ',' | sed 's/,$//')

if [ -n "$PIDS" ]; then
  # 采样 CPU/内存
  for i in $(seq 1 3); do
    ps -p $(echo $PIDS | tr ',' ' ') -o pid,%cpu,%mem,rss,comm --no-headers 2>/dev/null >> "$TMPDIR/ps_samples.txt"
    sleep $(( DURATION / 3 ))
  done
fi

wait  # 等待 hz 测量完成

echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="

echo ""
echo "--- Topic Publish Rates ---"
printf "%-22s %s\n" "TOPIC" "RATE"
echo "--------------------------------------"
for key in "${!TOPICS[@]}"; do
  rate="N/A"
  if [ -f "$TMPDIR/$key.txt" ] && [ -s "$TMPDIR/$key.txt" ]; then
    rate=$(cat "$TMPDIR/$key.txt" | grep -oP 'average rate: \K[0-9.]+' || echo "N/A")
    if [ "$rate" != "N/A" ]; then
      rate="${rate} Hz"
    fi
  fi
  printf "%-22s %s\n" "$key" "$rate"
done

echo ""
echo "--- Node CPU / Memory ---"
printf "%-25s %8s %8s %10s\n" "PROCESS" "CPU%" "MEM%" "RSS(MB)"
echo "------------------------------------------------------"
if [ -f "$TMPDIR/ps_samples.txt" ]; then
  # 按进程名取平均
  awk '{
    cpu[$5]+=$2; mem[$5]+=$3; rss[$5]+=$4; count[$5]++
  } END {
    for (name in cpu) {
      printf "%-25s %7.1f%% %7.1f%% %9.1f\n", name, cpu[name]/count[name], mem[name]/count[name], rss[name]/count[name]/1024
    }
  }' "$TMPDIR/ps_samples.txt" | sort -t% -k2 -rn
fi

echo ""
echo "--- Node Timing (check terminal logs) ---"
echo "semantic_cloud_node: [perf] cvt=?ms yolo=?ms cloud=?ms ..."
echo "semantic_map_node:   [perf] map: merge=?ms voxel=?ms ..."
echo "object_map_node:     [perf] obj: extract=?ms publish=?ms ..."
echo ""
echo "Tip: 在系统运行终端查看 [perf] 行获取每个节点的详细耗时"

# 总内存
echo ""
echo "--- System Memory ---"
free -h | head -2

echo ""
echo "--- GPU Usage ---"
if command -v tegrastats &>/dev/null; then
  timeout 2 tegrastats --interval 1000 2>/dev/null | head -1
elif command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
else
  echo "No GPU monitor available"
fi

rm -rf "$TMPDIR"

echo ""
echo "=========================================="
echo "  Profiling complete"
echo "=========================================="
