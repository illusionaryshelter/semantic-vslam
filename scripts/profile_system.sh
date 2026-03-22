#!/bin/bash
# =============================================================
# profile_system.sh — 语义 SLAM 系统性能 Profiling
#
# 用法:
#   1. 启动系统: ros2 launch semantic_vslam semantic_slam.launch.py ...
#   2. 另开终端: bash scripts/profile_system.sh [duration_seconds]
# =============================================================

DURATION=${1:-15}
echo "=========================================="
echo "  Semantic SLAM Profiler"
echo "  Duration: ${DURATION}s"
echo "=========================================="

source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null

TMPDIR=$(mktemp -d)

# ---- 1. Topic Rates ----
echo ""
echo "--- [1/3] Topic Rates (测量 ${DURATION}s) ---"

TOPICS=(
  "/camera/color/image_raw"
  "/camera/depth/image_raw"
  "/semantic_vslam/semantic_cloud"
  "/semantic_vslam/label_map"
  "/semantic_vslam/semantic_map_cloud"
  "/semantic_vslam/grid_map"
  "/semantic_vslam/grid_map_visual"
  "/semantic_vslam/object_markers"
  "/odom"
)

for topic in "${TOPICS[@]}"; do
  fname=$(echo "$topic" | tr '/' '_')
  (timeout ${DURATION} ros2 topic hz "$topic" --window 30 2>&1 > "$TMPDIR/${fname}.txt") &
done

# ---- 2. CPU/Memory 采样 ----
echo "--- [2/3] CPU/Memory (采样中...) ---"
sleep 2

for i in $(seq 1 3); do
  ps aux | grep -E 'semantic_cloud|semantic_map|object_map|rtabmap|rgbd_odometry' | grep -v grep | \
    awk '{printf "%-30s %6s %6s %8s\n", $11, $3, $4, $6}' >> "$TMPDIR/ps_samples.txt"
  sleep $(( (DURATION - 2) / 3 ))
done

wait  # 等待 hz 完成

# ---- 输出结果 ----
echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="

echo ""
echo "--- Topic Publish Rates ---"
printf "%-40s %s\n" "TOPIC" "RATE"
echo "------------------------------------------------------------"
for topic in "${TOPICS[@]}"; do
  fname=$(echo "$topic" | tr '/' '_')
  file="$TMPDIR/${fname}.txt"
  rate="(no data)"
  if [ -f "$file" ] && [ -s "$file" ]; then
    # ros2 topic hz 输出: "average rate: 29.998"
    extracted=$(grep -m1 'average rate' "$file" | sed 's/.*average rate: //' | awk '{printf "%.1f Hz", $1}')
    if [ -n "$extracted" ]; then
      rate="$extracted"
    fi
  fi
  printf "%-40s %s\n" "$topic" "$rate"
done

echo ""
echo "--- Node CPU / Memory (average of 3 samples) ---"
printf "%-35s %8s %8s %10s\n" "PROCESS" "CPU%" "MEM%" "RSS(MB)"
echo "------------------------------------------------------------"
if [ -f "$TMPDIR/ps_samples.txt" ]; then
  awk '{
    cpu[$1]+=$2; mem[$1]+=$3; rss[$1]+=$4; count[$1]++
  } END {
    for (name in cpu) {
      printf "%-35s %7.1f%% %7.1f%% %9.1f\n", name, cpu[name]/count[name], mem[name]/count[name], rss[name]/count[name]/1024
    }
  }' "$TMPDIR/ps_samples.txt" | sort -t% -k2 -rn
fi

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
  echo "No GPU monitor available (install tegrastats or nvidia-smi)"
fi

echo ""
echo "--- [perf] Logs ---"
echo "要查看节点内部详细耗时, 请在 params.yaml 中设置 enable_profiling: true"
echo "重启系统后在启动终端查看 [perf] 行"

rm -rf "$TMPDIR"
echo ""
echo "=========================================="
echo "  Profiling complete"
echo "=========================================="
