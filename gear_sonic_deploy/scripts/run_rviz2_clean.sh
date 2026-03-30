#!/usr/bin/env bash
# Launch rviz2 (and other ROS 2 GUI libs) without Conda's older libstdc++ winning.
#
# Error this fixes:
#   librclcpp.so: version `GLIBCXX_3.4.30' not found (required by ... libstdc++.so.6 from conda)
#
# Usage:
#   ./gear_sonic_deploy/scripts/run_rviz2_clean.sh
#   ./gear_sonic_deploy/scripts/run_rviz2_clean.sh -d /path/to/config.rviz
#
# Or one-liner (prepend system lib dir so linker picks newer libstdc++):
#   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
#   source /opt/ros/humble/setup.bash && rviz2
#
set -eo pipefail

ARCH="$(uname -m)"
SYS_LIB="/usr/lib/${ARCH}-linux-gnu"
if [[ ! -d "$SYS_LIB" ]]; then
  SYS_LIB="/usr/lib64"
fi
LIB_ROOT="/lib/${ARCH}-linux-gnu"
[[ -d "$LIB_ROOT" ]] || LIB_ROOT="/lib64"

export LD_LIBRARY_PATH="${SYS_LIB}:${LIB_ROOT}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  # shellcheck source=/dev/null
  source "$ROS_SETUP"
else
  echo "Warning: not found: $ROS_SETUP (set ROS_SETUP)" >&2
fi

exec rviz2 "$@"
