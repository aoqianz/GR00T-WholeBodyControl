#!/usr/bin/env python3
"""
Bridge: TeleVuer ZMQ (teleop) -> GEAR-SONIC deploy ZMQ (zmq_manager).

Subscribes to TeleVuer's JSON multipart stream (same as test_tv_wrapper_zmq_client.py):
  [topic_bytes, json_bytes]
Payload: head_pose / left_wrist_pose / right_wrist_pose -> "matrix" (4x4).
Head orientation in ``vr_orientation`` is forced to upright identity (wxyz); hands still track VR.
Head position still follows the headset (same frame as wrists); only head rotation is held fixed.

Republishes Sonic packed messages (same wire format as test_zmq_manager.py) on TCP:
  - topic "command": start / stop / planner mode
  - topic "planner": mode, movement, facing, speed, height, vr_position[9], vr_orientation[12]

When TeleVuer JSON has "mode": "controller", by default reads Quest thumbsticks from the same
payload as televuer/zmq_wrapper.py publishes:
  - left_ctrl_thumbstickValue  [x, y]  (normalized; +x right, +y back)
  - right_ctrl_thumbstickValue [x, y]  (right stick X used as turn rate)
Left stick -> robot horizontal: robot x += -joystick_y, robot y += -joystick_x (then rotate by
integrated facing into world movement). Right stick X still applies yaw (inverted from raw stick).
Use --disable-joystick-locomotion to drive upper body only (IDLE legs).

Prerequisites:
  - TeleVuerZmqWrapper publishing (e.g. teleop/televuer/example/test_tv_wrapper_zmq.py)
  - g1_deploy_onnx_ref --input-type zmq_manager --zmq-host <this machine>

Usage:
  export PYTHONPATH=/path/to/teleop/televuer/src:$PYTHONPATH
  python3 televr_sonic_zmq_bridge.py --sonic-host '*' --sonic-port 5556 \\
      --teleop-ipc /tmp/televuer_teledata.sock --rate-hz 60

Start deploy *before* or right after this binds; command messages are resent for the first
few seconds so ZMQ does not drop the only start pulse (slow-joiner).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HEADER_SIZE = 1280  # ZMQPackedMessageSubscriber::HEADER_SIZE

# LocomotionMode (localmotion_kplanner.hpp)
LOCOMOTION_IDLE = 0
LOCOMOTION_SLOW_WALK = 1
LOCOMOTION_WALK = 2

# Head quaternion in vr_orientation (w, x, y, z): upright, no tilt vs parent frame.
HEAD_QUAT_UPRIGHT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _ensure_televuer_client():
    try:
        from televuer import TeleDataZmqClient  # type: ignore

        return TeleDataZmqClient
    except ImportError:
        pass
    candidates = [
        os.environ.get("TELEVUER_SRC"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "teleop", "televuer", "src"),
        os.path.expanduser("~/teleop/televuer/src"),
        "/home/aoqian/teleop/televuer/src",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
    try:
        from televuer import TeleDataZmqClient  # type: ignore

        return TeleDataZmqClient
    except ImportError as e:
        raise ImportError(
            "Cannot import TeleDataZmqClient. pip install televuer or set PYTHONPATH to "
            "teleop/televuer/src (or TELEVUER_SRC)."
        ) from e


def _rotmat_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    q /= n
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _matrix_from_payload(entry: Dict[str, Any]) -> np.ndarray:
    m = entry.get("matrix")
    if m is None:
        raise ValueError("pose entry missing 'matrix'")
    return np.asarray(m, dtype=np.float64).reshape(4, 4)


def parse_controller_thumbsticks(payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Read Quest / Touch thumbsticks from TeleVuer ZMQ JSON (controller branch of
    televuer/zmq_wrapper.py TeleDataZmqPublisher._build_payload).

    Fields:
      left_ctrl_thumbstickValue:  [x, y] float, normalized (+x right, +y back)
      right_ctrl_thumbstickValue: [x, y] float, normalized

    Returns:
      left_xy, right_xy as float32 length-2 vectors; has_controller True if mode == "controller".
    """
    if payload.get("mode") != "controller":
        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), False

    def _vec2(key: str) -> np.ndarray:
        v = payload.get(key)
        if v is None or not isinstance(v, (list, tuple)) or len(v) < 2:
            return np.zeros(2, dtype=np.float32)
        return np.clip(np.asarray([float(v[0]), float(v[1])], dtype=np.float32), -1.0, 1.0)

    left = _vec2("left_ctrl_thumbstickValue")
    right = _vec2("right_ctrl_thumbstickValue")
    return left, right, True


def tele_payload_to_vr_three_point(payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    TeleVuerWrapper JSON -> vr_position (9,) and vr_orientation (12,).
    Order: left wrist xyz, right wrist xyz, head xyz; quats wxyz x3.

    Head orientation is not taken from VR: it is fixed to ``HEAD_QUAT_UPRIGHT_WXYZ`` so teleop
    drives hand poses only for rotation; the robot head target stays level/upright in frame.
    """
    head = _matrix_from_payload(payload["head_pose"])
    left = _matrix_from_payload(payload["left_wrist_pose"])
    right = _matrix_from_payload(payload["right_wrist_pose"])

    vr_position = np.concatenate(
        [left[:3, 3], right[:3, 3], head[:3, 3]]
    ).astype(np.float32)

    lq = _rotmat_to_quat_wxyz(left[:3, :3])
    rq = _rotmat_to_quat_wxyz(right[:3, :3])
    hq = HEAD_QUAT_UPRIGHT_WXYZ
    vr_orientation = np.array(lq + rq + hq, dtype=np.float32)

    return vr_position, vr_orientation


def joystick_to_planner_locomotion(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    facing_yaw: float,
    dt: float,
    *,
    dead_zone: float = 0.12,
    yaw_scale: float = 1.8,
    walk_stick_mag: float = 0.72,
) -> Tuple[int, np.ndarray, np.ndarray, float, float, float]:
    """
    Map TeleVuer thumbsticks to Sonic planner MovementState (mode, movement, facing, speed, height).

    Joystick axes (+x right, +y back). Robot body horizontal (x forward, y left) before world yaw:
      v_body_x = -joystick_y   (push stick forward -> -y -> +robot x)
      v_body_y = -joystick_x   (push stick left -> -x -> +robot y)
    Then rotate by integrated facing_yaw into world XY for ``movement`` (unit direction).
    Right-stick X uses negated raw axis for yaw rate (prior teleop convention).
    """
    jx, jy = float(left_xy[0]), float(left_xy[1])
    rx = -float(right_xy[0])
    ry = float(right_xy[1])  # reserved (e.g. future strafe trim); TeleVuer publishes 2D

    facing_yaw = facing_yaw + rx * yaw_scale * dt
    c = float(np.cos(facing_yaw))
    s = float(np.sin(facing_yaw))
    facing = np.array([c, s, 0.0], dtype=np.float32)

    vx_body = -jy
    vy_body = -jx
    mag = float(np.hypot(vx_body, vy_body))
    if mag < dead_zone:
        return (
            LOCOMOTION_IDLE,
            np.zeros(3, dtype=np.float32),
            facing,
            -1.0,
            -1.0,
            facing_yaw,
        )

    # Body (x fwd, y left) -> world
    mx = vx_body * c - vy_body * s
    my = vx_body * s + vy_body * c
    movement = np.array([mx, my, 0.0], dtype=np.float32)
    n = float(np.linalg.norm(movement))
    if n > 1e-6:
        movement = (movement / n).astype(np.float32)

    # Magnitude -> SLOW_WALK with explicit speed, or WALK with default speed
    mag_n = min(1.0, (mag - dead_zone) / max(1e-6, 1.0 - dead_zone))
    if mag < walk_stick_mag:
        mode = LOCOMOTION_SLOW_WALK
        # Map to planner speed range similar in spirit to ROS2 slow-walk binning
        speed = float(0.22 + 0.55 * mag_n)
    else:
        mode = LOCOMOTION_WALK
        speed = -1.0

    _ = ry  # silence unused in case we extend later
    return int(mode), movement, facing, speed, -1.0, facing_yaw


class SonicZmqPublisher:
    def __init__(self, host: str, port: int, verbose: bool = False):
        import zmq

        self._zmq = zmq
        self.verbose = verbose
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        ep = f"tcp://{host}:{port}"
        self.socket.bind(ep)
        time.sleep(0.5)
        if verbose:
            print(f"[SonicPublisher] Bound {ep}")

    def send_command(self, start: bool, stop: bool, planner: bool) -> None:
        topic = b"command"
        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
        header_json = json.dumps(header).encode("utf-8")
        header_bytes = header_json + b"\x00" * (HEADER_SIZE - len(header_json))
        data = struct.pack("BBB", int(start), int(stop), int(planner))
        self.socket.send(topic + header_bytes + data)

    def send_planner(
        self,
        mode: int,
        movement: np.ndarray,
        facing: np.ndarray,
        speed: float = -1.0,
        height: float = -1.0,
        vr_position: Optional[np.ndarray] = None,
        vr_orientation: Optional[np.ndarray] = None,
    ) -> None:
        topic = b"planner"
        fields: List[Dict[str, Any]] = [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ]
        if vr_position is not None:
            fields.append({"name": "vr_position", "dtype": "f32", "shape": [9]})
        if vr_orientation is not None:
            fields.append({"name": "vr_orientation", "dtype": "f32", "shape": [12]})

        header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
        header_json = json.dumps(header).encode("utf-8")
        header_bytes = header_json + b"\x00" * (HEADER_SIZE - len(header_json))

        data = b""
        data += struct.pack("<i", int(mode))
        data += np.asarray(movement, dtype=np.float32).reshape(3).tobytes()
        data += np.asarray(facing, dtype=np.float32).reshape(3).tobytes()
        data += struct.pack("<ff", float(speed), float(height))
        if vr_position is not None:
            data += np.asarray(vr_position, dtype=np.float32).reshape(9).tobytes()
        if vr_orientation is not None:
            data += np.asarray(vr_orientation, dtype=np.float32).reshape(12).tobytes()

        self.socket.send(topic + header_bytes + data)

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleVuer IPC -> Sonic ZMQ bridge")
    parser.add_argument("--sonic-host", type=str, default="*", help="PUB bind address")
    parser.add_argument("--sonic-port", type=int, default=5556)
    parser.add_argument("--teleop-ipc", type=str, default="/tmp/televuer_teledata.sock")
    parser.add_argument("--teleop-topic", type=str, default="televuer.teledata")
    parser.add_argument("--teleop-timeout-ms", type=int, default=5)
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--disable-joystick-locomotion",
        action="store_true",
        help="Do not map Quest thumbsticks to planner legs; upper body still follows VR poses",
    )
    parser.add_argument(
        "--loco-from-sticks",
        action="store_true",
        help="(deprecated) Joystick locomotion is on by default in controller mode",
    )
    parser.add_argument("--stick-dead-zone", type=float, default=0.12)
    parser.add_argument("--stick-yaw-scale", type=float, default=1.8)
    parser.add_argument(
        "--stick-walk-threshold",
        type=float,
        default=0.72,
        help="Left stick magnitude above this uses WALK; below uses SLOW_WALK with scaled speed",
    )
    parser.add_argument(
        "--command-resend-sec",
        type=float,
        default=5.0,
        help="Resend start+planner command every 0.5s for this many seconds (ZMQ slow-joiner)",
    )
    args = parser.parse_args()

    TeleDataZmqClient = _ensure_televuer_client()
    tele_client = TeleDataZmqClient(ipc_path=args.teleop_ipc, topic=args.teleop_topic)
    sonic = SonicZmqPublisher(host=args.sonic_host, port=args.sonic_port, verbose=True)

    t_start = time.time()
    period = 1.0 / max(args.rate_hz, 1.0)
    facing_yaw = 0.0
    last_t = time.time()
    last_cmd_t = 0.0
    last_vrp: Optional[np.ndarray] = None
    last_vro: Optional[np.ndarray] = None
    last_teleop_mode: Optional[str] = None
    last_left_stick = np.zeros(2, dtype=np.float32)
    last_right_stick = np.zeros(2, dtype=np.float32)

    sonic.send_command(start=True, stop=False, planner=True)
    last_cmd_t = time.time()
    if args.verbose:
        print("[Bridge] initial command: start=True planner=True")

    try:
        while True:
            t0 = time.time()

            # Resend start+planner while deploy may still be connecting (ZMQ slow-joiner)
            if t0 - t_start < args.command_resend_sec and (t0 - last_cmd_t) >= 0.5:
                sonic.send_command(start=True, stop=False, planner=True)
                last_cmd_t = t0
                if args.verbose:
                    print("[Bridge] command: start=True planner=True")

            payload = tele_client.recv_payload(timeout_ms=args.teleop_timeout_ms)
            dt = max(t0 - last_t, 1e-4)
            last_t = t0

            if payload is not None:
                m = payload.get("mode")
                if isinstance(m, str):
                    last_teleop_mode = m
                left_s, right_s, _ = parse_controller_thumbsticks(payload)
                last_left_stick = left_s
                last_right_stick = right_s
                try:
                    last_vrp, last_vro = tele_payload_to_vr_three_point(payload)
                except (KeyError, ValueError) as e:
                    if args.verbose:
                        print(f"[Bridge] bad teleop JSON: {e}")

            if last_vrp is None or last_vro is None:
                time.sleep(period)
                continue

            use_joystick = (
                not args.disable_joystick_locomotion and last_teleop_mode == "controller"
            )
            if use_joystick:
                mode, movement, facing, speed, height, facing_yaw = joystick_to_planner_locomotion(
                    last_left_stick,
                    last_right_stick,
                    facing_yaw,
                    dt,
                    dead_zone=args.stick_dead_zone,
                    yaw_scale=args.stick_yaw_scale,
                    walk_stick_mag=args.stick_walk_threshold,
                )
            else:
                mode = LOCOMOTION_IDLE
                movement = np.zeros(3, dtype=np.float32)
                facing = np.array(
                    [np.cos(facing_yaw), np.sin(facing_yaw), 0.0], dtype=np.float32
                )
                speed = -1.0
                height = -1.0

            sonic.send_planner(
                mode,
                movement,
                facing,
                speed=speed,
                height=height,
                vr_position=last_vrp,
                vr_orientation=last_vro,
            )

            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\n[Bridge] stop command...")
        sonic.send_command(start=False, stop=True, planner=True)
    finally:
        tele_client.close()
        sonic.close()


if __name__ == "__main__":
    main()
