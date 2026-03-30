#!/usr/bin/env python3
"""
TeleVuer IPC (same JSON as televr_sonic_zmq_bridge) -> 3D visualization.

Backends:
  rviz (default) — ROS 2 TF + MarkerArray for RViz2 (needs rclpy; avoid conda Python, see below).
  matplotlib — 3D scatter window; no ROS (must install matplotlib *into the same Python* as this script).
  tty — print xyz to terminal only; no matplotlib / no ROS (good when conda lacks matplotlib).

TeleVuer JSON ``ts`` supports ``--print-latency`` (receive wall time - ts, ms).

Dependencies (this script): **numpy**, **pyzmq** in the same Python you use to run it
(``pip install numpy pyzmq`` or ``conda install numpy pyzmq``). Optional: matplotlib, ROS 2 rclpy.

ROS + conda: if rclpy fails with GLIBCXX / libstdc++.so.6, conda’s libstdc++ is too old.
  Fix A: ``/usr/bin/python3`` (after ``source /opt/ros/humble/setup.bash``) + pip install numpy pyzmq into user/venv.
  Fix B: ``conda install -c conda-forge 'libstdcxx-ng>=12'`` then retry.
  Fix C: use ``--backend matplotlib`` (no ROS).

Televuer + RViz (two terminals):
  Terminal A:  cd teleop/televuer && ./run.sh
  Terminal B:  ./gear_sonic_deploy/scripts/run_televr_rviz_viz.sh
  (Or manually: source /opt/ros/humble/setup.bash && /usr/bin/python3 .../televr_rviz_three_point_viz.py --backend rviz)

Example (matplotlib — use pip in *active* env: pip install matplotlib):
  python3 .../televr_rviz_three_point_viz.py --backend matplotlib --print-latency

Example (tty — only numpy + pyzmq; no televuer/vuer on PYTHONPATH):
  python3 .../televr_rviz_three_point_viz.py --backend tty --print-latency
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HEAD_QUAT_UPRIGHT_WXYZ = (1.0, 0.0, 0.0, 0.0)


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


def _ipc_zmq_endpoint(ipc_path: str) -> str:
    return "ipc://" + os.path.abspath(os.path.expanduser(ipc_path))


class TeleVuerTeledataZmqSubscriber:
    """
    SUB socket + JSON decode; same wire format as televuer.zmq_wrapper.TeleDataZmqClient.

    Implemented here so we do not import the ``televuer`` package (its __init__ pulls in ``vuer``).
    """

    def __init__(
        self,
        ipc_path: str = "/tmp/televuer_teledata.sock",
        topic: str = "televuer.teledata",
        rcv_hwm: int = 1,
        linger_ms: int = 0,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError(
                "pyzmq is not installed for this Python interpreter.\n"
                f"  Interpreter: {sys.executable}\n"
                "  Install:  pip install pyzmq\n"
                "  Or:       conda install pyzmq\n"
                "  (Debian system Python: sudo apt install python3-zmq)"
            ) from exc

        self._zmq = zmq
        self._topic_bytes = topic.encode("utf-8")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, rcv_hwm)
        self._socket.setsockopt(zmq.LINGER, linger_ms)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic_bytes)
        self._socket.connect(_ipc_zmq_endpoint(ipc_path))

    def recv_payload(self, timeout_ms: Optional[int] = 1000) -> Optional[Dict[str, Any]]:
        if timeout_ms is not None:
            poller = self._zmq.Poller()
            poller.register(self._socket, self._zmq.POLLIN)
            events = dict(poller.poll(timeout_ms))
            if self._socket not in events:
                return None

        topic_bytes, payload_bytes = self._socket.recv_multipart()
        if topic_bytes != self._topic_bytes:
            return None
        return json.loads(payload_bytes.decode("utf-8"))

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None


def poses_from_payload(
    payload: Dict[str, Any], head_upright_like_bridge: bool
) -> List[Tuple[str, np.ndarray, Tuple[float, float, float, float]]]:
    head = _matrix_from_payload(payload["head_pose"])
    left = _matrix_from_payload(payload["left_wrist_pose"])
    right = _matrix_from_payload(payload["right_wrist_pose"])

    lq = _rotmat_to_quat_wxyz(left[:3, :3])
    rq = _rotmat_to_quat_wxyz(right[:3, :3])
    if head_upright_like_bridge:
        hq = HEAD_QUAT_UPRIGHT_WXYZ
    else:
        hq = _rotmat_to_quat_wxyz(head[:3, :3])

    return [
        ("vr_left_wrist", left, lq),
        ("vr_right_wrist", right, rq),
        ("vr_head", head, hq),
    ]


def _rclpy_import_error_message(exc: BaseException) -> str:
    msg = str(exc)
    lines = [
        "Failed to import ROS 2 rclpy. General fix: source /opt/ros/humble/setup.bash",
        "Packages: ros-humble-rclpy ros-humble-tf2-ros ros-humble-visualization-msgs",
    ]
    if "GLIBCXX" in msg or "libstdc++" in msg:
        lines = [
            "Conda (or another env) is providing an older libstdc++.so.6 than ROS 2’s rclpy needs.",
            "  1) Run RViz backend with system Python: /usr/bin/python3 this_script.py ...",
            "  2) Or: conda install -c conda-forge 'libstdcxx-ng>=12'",
            "  3) Or skip ROS: --backend matplotlib",
            f"Underlying error: {msg}",
        ]
    else:
        lines.append(f"Underlying error: {msg}")
    return "\n".join(lines)


def _try_import_rclpy():
    try:
        import rclpy  # noqa: F401
        from geometry_msgs.msg import Quaternion, TransformStamped
        from rclpy.node import Node
        from std_msgs.msg import ColorRGBA
        from tf2_ros import TransformBroadcaster
        from visualization_msgs.msg import Marker, MarkerArray

        return (
            rclpy,
            Node,
            TransformBroadcaster,
            MarkerArray,
            Marker,
            TransformStamped,
            Quaternion,
            ColorRGBA,
        )
    except ImportError as e:
        raise ImportError(_rclpy_import_error_message(e)) from e


def _wxyz_to_geometry_quat(w: float, x: float, y: float, z: float, QuatCls):
    q = QuatCls()
    q.x = float(x)
    q.y = float(y)
    q.z = float(z)
    q.w = float(w)
    return q


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TeleVuer ZMQ IPC -> RViz2 or matplotlib 3-point viz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backend",
        choices=("rviz", "matplotlib", "tty"),
        default="rviz",
        help="rviz: ROS2. matplotlib: 3D window (pip install matplotlib in this env). tty: stdout only.",
    )
    p.add_argument("--teleop-ipc", type=str, default="/tmp/televuer_teledata.sock")
    p.add_argument("--teleop-topic", type=str, default="televuer.teledata")
    p.add_argument("--teleop-timeout-ms", type=int, default=2)
    p.add_argument("--fixed-frame", type=str, default="vr_world")
    p.add_argument("--markers-topic", type=str, default="vr_three_point/markers")
    p.add_argument("--sphere-scale", type=float, default=0.05, help="RViz marker sphere size (m)")
    p.add_argument(
        "--head-upright-like-bridge",
        action="store_true",
        help="Force head orientation to identity (matches televr_sonic_zmq_bridge)",
    )
    p.add_argument("--print-latency", action="store_true")
    p.add_argument("--timer-ms", type=int, default=4, help="ROS timer period (ms)")
    p.add_argument(
        "--matplotlib-pause",
        type=float,
        default=0.01,
        help="matplotlib loop sleep (s); lower = higher poll rate",
    )
    p.add_argument(
        "--tty-rate-hz",
        type=float,
        default=20.0,
        help="tty backend: max print lines per second (position updates)",
    )
    return p


def _latency_tick(args, payload: Dict[str, Any], state: Dict[str, Any]) -> None:
    if not args.print_latency or "ts" not in payload:
        return
    try:
        ts = float(payload["ts"])
        dt_ms = (time.time() - ts) * 1000.0
        state["lat_sum"] = state.get("lat_sum", 0.0) + dt_ms
        state["lat_n"] = state.get("lat_n", 0) + 1
    except (TypeError, ValueError):
        return
    t_wall = time.time()
    if t_wall - state.get("last_print", 0.0) >= 1.0 and state.get("lat_n", 0) > 0:
        mean_ms = state["lat_sum"] / state["lat_n"]
        print(
            f"[latency] VR sample age (recv_wall - payload ts): mean={mean_ms:.1f} ms "
            f"over {state['lat_n']} msgs",
            flush=True,
        )
        state["lat_sum"] = 0.0
        state["lat_n"] = 0
        state["last_print"] = t_wall


class TelevrThreePointVizNode:
    def __init__(self, args: argparse.Namespace) -> None:
        (
            rclpy,
            Node,
            TransformBroadcaster,
            MarkerArray,
            Marker,
            TransformStamped,
            Quaternion,
            ColorRGBA,
        ) = _try_import_rclpy()

        self._rclpy = rclpy
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self._TransformStamped = TransformStamped
        self._ColorRGBA = ColorRGBA
        self._Quaternion = Quaternion
        self.args = args

        rclpy.init()
        self.node = Node("televr_three_point_viz")
        self._tf_broadcaster = TransformBroadcaster(self.node)
        self._marker_pub = self.node.create_publisher(MarkerArray, args.markers_topic, 10)

        self._tele = TeleVuerTeledataZmqSubscriber(ipc_path=args.teleop_ipc, topic=args.teleop_topic)

        self._lat_state: Dict[str, Any] = {"last_print": time.time()}

        self._timer = self.node.create_timer(args.timer_ms / 1000.0, self._on_timer)
        self.node.get_logger().info(
            f"TeleVuer IPC {args.teleop_ipc} topic={args.teleop_topic} -> "
            f"TF frame={args.fixed_frame} markers={args.markers_topic}"
        )

    def _on_timer(self) -> None:
        args = self.args
        payload = self._tele.recv_payload(timeout_ms=args.teleop_timeout_ms)
        if payload is None:
            return

        now = self.node.get_clock().now().to_msg()
        fixed = args.fixed_frame

        try:
            items = poses_from_payload(payload, args.head_upright_like_bridge)
        except (KeyError, ValueError) as e:
            self.node.get_logger().warning(f"Bad teleop payload: {e}")
            return

        transforms = []
        markers = self._MarkerArray()
        marker_id = 0
        colors = [
            self._ColorRGBA(r=0.2, g=0.9, b=0.3, a=1.0),
            self._ColorRGBA(r=0.3, g=0.5, b=1.0, a=1.0),
            self._ColorRGBA(r=1.0, g=0.35, b=0.2, a=1.0),
        ]

        for i, (child, T, quat_wxyz) in enumerate(items):
            t = self._TransformStamped()
            t.header.stamp = now
            t.header.frame_id = fixed
            t.child_frame_id = child
            t.transform.translation.x = float(T[0, 3])
            t.transform.translation.y = float(T[1, 3])
            t.transform.translation.z = float(T[2, 3])
            t.transform.rotation = _wxyz_to_geometry_quat(
                quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], self._Quaternion
            )
            transforms.append(t)

            m = self._Marker()
            m.header.stamp = now
            m.header.frame_id = fixed
            m.ns = "vr_three_point"
            m.id = marker_id
            marker_id += 1
            m.type = self._Marker.SPHERE
            m.action = self._Marker.ADD
            m.pose.position.x = float(T[0, 3])
            m.pose.position.y = float(T[1, 3])
            m.pose.position.z = float(T[2, 3])
            m.pose.orientation = t.transform.rotation
            m.scale.x = args.sphere_scale
            m.scale.y = args.sphere_scale
            m.scale.z = args.sphere_scale
            m.color = colors[i]
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            markers.markers.append(m)

        self._tf_broadcaster.sendTransform(transforms)
        self._marker_pub.publish(markers)

        _latency_tick(args, payload, self._lat_state)

    def run(self) -> None:
        try:
            self._rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self._tele.close()
            self.node.destroy_node()
            self._rclpy.shutdown()


def run_matplotlib(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        exe = sys.executable
        raise ImportError(
            "matplotlib is not installed for the Python you are running.\n"
            f"  This interpreter: {exe}\n"
            "  apt/system matplotlib (e.g. /usr/lib/python3/dist-packages) is NOT visible to conda envs.\n"
            "  Fix:  pip install matplotlib\n"
            "  Or:    conda install matplotlib\n"
            "  Or use: --backend tty   (no matplotlib)\n"
            "  Or run with: /usr/bin/python3 ...  if system has python3-matplotlib"
        ) from e

    tele = TeleVuerTeledataZmqSubscriber(ipc_path=args.teleop_ipc, topic=args.teleop_topic)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("TeleVuer 3-point (green=left, blue=right, orange=head)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    sc = ax.scatter(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        c=["g", "b", "#ff6600"],
        s=[60, 60, 80],
        depthshade=True,
    )
    lat_state: Dict[str, Any] = {"last_print": time.time()}

    def autoscale_from_xyz(xs: List[float], ys: List[float], zs: List[float]) -> None:
        cx = sum(xs) / 3.0
        cy = sum(ys) / 3.0
        cz = sum(zs) / 3.0
        span = max(0.4, max(max(abs(v - cx) for v in xs), max(abs(v - cy) for v in ys), max(abs(v - cz) for v in zs)) * 2.5)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    print(
        f"[matplotlib] IPC {args.teleop_ipc} topic={args.teleop_topic} — close window to exit.",
        flush=True,
    )

    try:
        while plt.fignum_exists(fig.number):
            payload = tele.recv_payload(timeout_ms=args.teleop_timeout_ms)
            if payload is not None:
                try:
                    items = poses_from_payload(payload, args.head_upright_like_bridge)
                except (KeyError, ValueError) as e:
                    print(f"[matplotlib] bad payload: {e}", flush=True)
                else:
                    xs = [float(T[0, 3]) for _, T, _ in items]
                    ys = [float(T[1, 3]) for _, T, _ in items]
                    zs = [float(T[2, 3]) for _, T, _ in items]
                    sc._offsets3d = (xs, ys, zs)  # type: ignore[attr-defined]
                    autoscale_from_xyz(xs, ys, zs)
                    _latency_tick(args, payload, lat_state)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(args.matplotlib_pause)
    except KeyboardInterrupt:
        pass
    finally:
        tele.close()
        plt.close("all")


def run_tty(args: argparse.Namespace) -> None:
    tele = TeleVuerTeledataZmqSubscriber(ipc_path=args.teleop_ipc, topic=args.teleop_topic)
    period = 1.0 / max(float(args.tty_rate_hz), 0.5)
    lat_state: Dict[str, Any] = {"last_print": time.time()}
    next_print = 0.0
    print(
        f"[tty] IPC {args.teleop_ipc} topic={args.teleop_topic} (Ctrl+C to exit)\n"
        "  columns: Lx Ly Lz | Rx Ry Rz | Hx Hy Hz | age_ms (if payload has ts)",
        flush=True,
    )
    try:
        while True:
            payload = tele.recv_payload(timeout_ms=max(args.teleop_timeout_ms, 50))
            if payload is None:
                continue
            try:
                items = poses_from_payload(payload, args.head_upright_like_bridge)
            except (KeyError, ValueError) as e:
                print(f"[tty] bad payload: {e}", flush=True)
                continue

            now = time.time()
            if now < next_print:
                _latency_tick(args, payload, lat_state)
                continue
            next_print = now + period

            parts = []
            for name, T, _ in items:
                parts.append(f"{T[0,3]:+.3f} {T[1,3]:+.3f} {T[2,3]:+.3f}")
            line = " | ".join(parts)
            if "ts" in payload:
                try:
                    age_ms = (time.time() - float(payload["ts"])) * 1000.0
                    line += f" | age={age_ms:6.1f}ms"
                except (TypeError, ValueError):
                    pass
            print(line, flush=True)
            _latency_tick(args, payload, lat_state)
    except KeyboardInterrupt:
        print("", flush=True)
    finally:
        tele.close()


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Note: ignoring extra argv: {unknown}", file=sys.stderr)

    if args.backend == "matplotlib":
        run_matplotlib(args)
    elif args.backend == "tty":
        run_tty(args)
    else:
        viz = TelevrThreePointVizNode(args)
        viz.run()


if __name__ == "__main__":
    main()
