#!/usr/bin/env bash
# Visualize TeleVuer VR three-point poses in RViz2 while teleop/televuer/run.sh (or
# example/test_tv_wrapper_zmq.py) is publishing on the IPC socket.
#
# Defaults match televuer: /tmp/televuer_teledata.sock, topic televuer.teledata
#
# Usage:
#   Terminal A:  cd teleop/televuer && ./run.sh
#   Terminal B:  cd GR00T-WholeBodyControl && ./gear_sonic_deploy/scripts/run_televr_rviz_viz.sh
#
# ROS uses system Python by default (avoids conda + rclpy libstdc++ issues). Override:
#   TELEVR_VIZ_PYTHON=/path/to/python3 ./gear_sonic_deploy/scripts/run_televr_rviz_viz.sh
#
# That Python needs: numpy, pyzmq, and ROS 2 rclpy stack. Examples:
#   sudo apt install python3-numpy python3-zmq ros-humble-rviz2 ros-humble-rclpy ros-humble-tf2-ros ros-humble-visualization-msgs
#   /usr/bin/python3 -m pip install --user numpy pyzmq   # if apt packages missing
#
# Do not use ``set -u`` here: sourcing /opt/ros/*/setup.bash references optional vars
# (e.g. AMENT_TRACE_SETUP_FILES) that may be unset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_PY="${SCRIPT_DIR}/televr_rviz_three_point_viz.py"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  # shellcheck source=/dev/null
  source "$ROS_SETUP"
else
  echo "Warning: ROS not sourced (file not found: $ROS_SETUP). Set ROS_SETUP or source ROS manually." >&2
fi

PYTHON_BIN="${TELEVR_VIZ_PYTHON:-/usr/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

IPC="${TELEVR_VIZ_IPC:-/tmp/televuer_teledata.sock}"
TOPIC="${TELEVR_VIZ_TOPIC:-televuer.teledata}"

echo "TeleVuer -> RViz2  (IPC=$IPC  topic=$TOPIC  python=$PYTHON_BIN)"
echo "RViz2: Fixed Frame = vr_world"
echo "       Add → By topic → /vr_three_point/markers  (MarkerArray)"
echo "       TF display on (shows vr_left_wrist, vr_right_wrist, vr_head)"
echo "If \`rviz2\` fails with GLIBCXX / libstdc++ while conda is active, use:"
echo "       ./gear_sonic_deploy/scripts/run_rviz2_clean.sh"
echo ""

exec "$PYTHON_BIN" "$VIZ_PY" \
  --backend rviz \
  --teleop-ipc "$IPC" \
  --teleop-topic "$TOPIC" \
  "$@"
