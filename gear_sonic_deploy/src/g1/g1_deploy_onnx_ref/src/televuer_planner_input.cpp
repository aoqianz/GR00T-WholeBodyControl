/**
 * @file televuer_planner_input.cpp
 */

#include "input_interface/televuer_planner_input.hpp"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <limits>
#include <thread>

#include <nlohmann/json.hpp>
#include <zmq.h>

#include "input_interface/input_command.hpp"
#include "localmotion_kplanner.hpp"
#include "math_utils.hpp"

namespace {

std::string IpcEndpoint(const std::string& ipc_path) {
  namespace fs = std::filesystem;
  fs::path p = fs::absolute(fs::path(ipc_path));
  return std::string("ipc://") + p.string();
}

void RotmatToQuatWxyz(const double R[9], double& qw, double& qx, double& qy, double& qz) {
  const double t = R[0] + R[4] + R[8];
  double w, x, y, z;
  if (t > 0.0) {
    double s = 0.5 / std::sqrt(t + 1.0);
    w = 0.25 / s;
    x = (R[7] - R[5]) * s;
    y = (R[2] - R[6]) * s;
    z = (R[3] - R[1]) * s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    double s = 2.0 * std::sqrt(1.0 + R[0] - R[4] - R[8]);
    w = (R[7] - R[5]) / s;
    x = 0.25 * s;
    y = (R[1] + R[3]) / s;
    z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    double s = 2.0 * std::sqrt(1.0 + R[4] - R[0] - R[8]);
    w = (R[2] - R[6]) / s;
    x = (R[1] + R[3]) / s;
    y = 0.25 * s;
    z = (R[5] + R[7]) / s;
  } else {
    double s = 2.0 * std::sqrt(1.0 + R[8] - R[0] - R[4]);
    w = (R[3] - R[1]) / s;
    x = (R[2] + R[6]) / s;
    y = (R[5] + R[7]) / s;
    z = 0.25 * s;
  }
  double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n < 1e-12) {
    qw = 1.0;
    qx = qy = qz = 0.0;
    return;
  }
  qw = w / n;
  qx = x / n;
  qy = y / n;
  qz = z / n;
}

bool ParsePoseMatrix4(const nlohmann::json& entry, double T[16]) {
  if (!entry.contains("matrix") || !entry["matrix"].is_array()) return false;
  const auto& m = entry["matrix"];
  // TeleVuer zmq_wrapper._pose_to_dict: 4x4 ndarray -> .tolist() is nested [[row0],[row1],[row2],[row3]].
  if (m.size() == 16) {
    for (size_t i = 0; i < 16; ++i) {
      if (!m[i].is_number()) return false;
      T[i] = m[i].get<double>();
    }
    return true;
  }
  if (m.size() == 4 && m[0].is_array()) {
    for (size_t r = 0; r < 4; ++r) {
      if (!m[r].is_array() || m[r].size() < 4) return false;
      for (size_t c = 0; c < 4; ++c) {
        if (!m[r][c].is_number()) return false;
        T[r * 4 + c] = m[r][c].get<double>();
      }
    }
    return true;
  }
  return false;
}

void ReadStick2(const nlohmann::json& j, const char* key, double& ox, double& oy) {
  ox = oy = 0.0;
  if (!j.contains(key) || !j[key].is_array()) return;
  const auto& a = j[key];
  if (a.size() < 2) return;
  ox = std::max(-1.0, std::min(1.0, a[0].get<double>()));
  oy = std::max(-1.0, std::min(1.0, a[1].get<double>()));
}

}  // namespace

TeleVuerPlannerInput::TeleVuerPlannerInput(std::string ipc_path,
                                           std::string topic,
                                           bool disable_joystick_locomotion,
                                           bool zmq_verbose)
    : ipc_path_(std::move(ipc_path)),
      topic_(std::move(topic)),
      disable_joystick_locomotion_(disable_joystick_locomotion),
      zmq_verbose_(zmq_verbose) {
  type_ = InputType::NETWORK;
  zmq_ctx_ = zmq_ctx_new();
  if (!zmq_ctx_) {
    throw std::runtime_error("TeleVuerPlannerInput: zmq_ctx_new failed");
  }
  zmq_socket_ = zmq_socket(zmq_ctx_, ZMQ_SUB);
  if (!zmq_socket_) {
    zmq_ctx_destroy(zmq_ctx_);
    zmq_ctx_ = nullptr;
    throw std::runtime_error("TeleVuerPlannerInput: zmq_socket failed");
  }
  int one = 1;
  zmq_setsockopt(zmq_socket_, ZMQ_LINGER, &one, sizeof(one));
  int rcv_hwm = 1;
  zmq_setsockopt(zmq_socket_, ZMQ_RCVHWM, &rcv_hwm, sizeof(rcv_hwm));
  int rcv_timeout_ms = 100;
  zmq_setsockopt(zmq_socket_, ZMQ_RCVTIMEO, &rcv_timeout_ms, sizeof(rcv_timeout_ms));
  // No ZMQ_CONFLATE: multipart + conflate has been problematic on some stacks; match televr_rviz subscriber.
  zmq_setsockopt(zmq_socket_, ZMQ_SUBSCRIBE, topic_.data(), topic_.size());

  const std::string ep = IpcEndpoint(ipc_path_);
  if (zmq_connect(zmq_socket_, ep.c_str()) != 0) {
    std::cerr << "[TeleVuerPlannerInput] zmq_connect failed: " << zmq_strerror(zmq_errno()) << " ep=" << ep
              << std::endl;
    zmq_close(zmq_socket_);
    zmq_ctx_destroy(zmq_ctx_);
    zmq_socket_ = nullptr;
    zmq_ctx_ = nullptr;
    throw std::runtime_error("TeleVuerPlannerInput: zmq_connect failed");
  }
  // Let subscription filter propagate (ipc PUB/SUB slow-joiner edge cases).
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // ReadStdinChar() uses read(); without O_NONBLOCK the Input thread can block forever,
  // and G1Deploy::Stop() will hang on input_thread_ptr_->Wait().
  {
    const int stdin_flags = fcntl(STDIN_FILENO, F_GETFL);
    if (stdin_flags >= 0) {
      if (fcntl(STDIN_FILENO, F_SETFL, stdin_flags | O_NONBLOCK) != 0 && zmq_verbose_) {
        std::cerr << "[TeleVuerPlannerInput] fcntl(STDIN_FILENO, O_NONBLOCK): " << std::strerror(errno)
                  << std::endl;
      }
    } else if (zmq_verbose_) {
      std::cerr << "[TeleVuerPlannerInput] fcntl(STDIN_FILENO, F_GETFL): " << std::strerror(errno) << std::endl;
    }
  }

  std::cout << "[TeleVuerPlannerInput] SUB " << ep << " topic='" << topic_ << "' RCVHWM=1" << std::endl;
  std::cout << "  Joystick locomotion: " << (disable_joystick_locomotion_ ? "off" : "on (controller mode)")
            << std::endl;
}

TeleVuerPlannerInput::~TeleVuerPlannerInput() {
  if (zmq_socket_) {
    zmq_close(zmq_socket_);
    zmq_socket_ = nullptr;
  }
  if (zmq_ctx_) {
    zmq_ctx_destroy(zmq_ctx_);
    zmq_ctx_ = nullptr;
  }
}

void TeleVuerPlannerInput::update() {
  emergency_stop_ = false;
  report_temperature_flag_ = false;
  start_control_ = false;
  stop_control_ = false;

  char ch;
  while (ReadStdinChar(ch)) {
    switch (ch) {
      case 'o':
      case 'O':
        emergency_stop_ = true;
        std::cout << "[TeleVuerPlannerInput] EMERGENCY STOP (O/o)" << std::endl;
        break;
      case 'f':
      case 'F':
        report_temperature_flag_ = true;
        break;
      case 'g':
      case 'G':
        AdjustLeftHandCompliance(0.1);
        break;
      case 'h':
      case 'H':
        AdjustLeftHandCompliance(-0.1);
        break;
      case 'b':
      case 'B':
        AdjustRightHandCompliance(0.1);
        break;
      case 'v':
      case 'V':
        AdjustRightHandCompliance(-0.1);
        break;
      case 'x':
      case 'X':
        AdjustMaxCloseRatio(0.1);
        break;
      case 'c':
      case 'C':
        AdjustMaxCloseRatio(-0.1);
        break;
      case ']':
        start_control_ = true;
        std::cout << "[TeleVuerPlannerInput] ] Start control (same as keyboard; use if teledata not wired yet)"
                  << std::endl;
        break;
      default:
        break;
    }
  }

  DrainSocketAndApplyLatest();
}

void TeleVuerPlannerInput::DrainSocketAndApplyLatest() {
  std::string last_body;
  std::string last_topic;

  for (;;) {
    zmq_msg_t part0;
    zmq_msg_init(&part0);
    int r = zmq_msg_recv(&part0, zmq_socket_, ZMQ_DONTWAIT);
    if (r == -1) {
      zmq_msg_close(&part0);
      if (zmq_errno() == EAGAIN) break;
      if (zmq_verbose_) {
        std::cerr << "[TeleVuerPlannerInput] recv error: " << zmq_strerror(zmq_errno()) << std::endl;
      }
      break;
    }

    zmq_msg_t part1;
    zmq_msg_init(&part1);
    if (zmq_msg_recv(&part1, zmq_socket_, 0) == -1) {
      zmq_msg_close(&part0);
      zmq_msg_close(&part1);
      break;
    }

    last_topic.assign(static_cast<char*>(zmq_msg_data(&part0)), zmq_msg_size(&part0));
    last_body.assign(static_cast<char*>(zmq_msg_data(&part1)), zmq_msg_size(&part1));
    zmq_msg_close(&part0);
    zmq_msg_close(&part1);
  }

  if (last_body.empty()) {
    if (++no_teledata_log_counter_ >= 200) {
      no_teledata_log_counter_ = 0;
      std::cerr
          << "[TeleVuerPlannerInput] No teledata on " << IpcEndpoint(ipc_path_) << " (SUB prefix '" << topic_
          << "').\n"
          << "  ipc:// works only on ONE machine: TeleVuer (run.sh / test_tv_wrapper_zmq.py) binds this path "
             "where Python runs. g1_deploy must run on that same host (same /tmp), not on the robot/another PC.\n"
          << "  If TeleVuer is on your laptop and deploy on the robot, use --input-type zmq_manager + "
             "televr_sonic_zmq_bridge.py (TCP), not --input-type televuer.\n"
          << "  Also: open the VR/web session so the publisher loop can send poses; check TeleVuer logs for "
             "'ZMQ publish loop error'; try rm -f " << ipc_path_ << " if an old PUB crashed.\n"
          << "  Press ] here to arm start without a VR frame (for debugging).\n";
    }
    return;
  }

  // ZMQ SUB filter is prefix-based; the first part may equal the configured topic or be a longer string.
  const bool topic_ok =
      (!topic_.empty() && last_topic.size() >= topic_.size() &&
       last_topic.compare(0, topic_.size(), topic_) == 0);
  if (!topic_ok) {
    if (++no_teledata_log_counter_ >= 200) {
      no_teledata_log_counter_ = 0;
      std::cerr << "[TeleVuerPlannerInput] Ignoring frame: first multipart part is '" << last_topic
                << "' (expected prefix '" << topic_
                << "'). Use --televuer-topic to match TeleVuer's publisher." << std::endl;
    }
    return;
  }

  bool applied = false;
  if (ApplyJsonPayload(last_body)) {
    start_control_ = true;
    applied = true;
  } else if (zmq_verbose_) {
    std::cerr << "[TeleVuerPlannerInput] ApplyJsonPayload failed (parse error or missing "
                 "head_pose/left_wrist_pose/right_wrist_pose matrix)."
              << std::endl;
  }

  if (applied) {
    no_teledata_log_counter_ = 0;
  } else if (++no_teledata_log_counter_ >= 200) {
    no_teledata_log_counter_ = 0;
    std::cerr << "[TeleVuerPlannerInput] Teledata received but ApplyJsonPayload failed (schema mismatch?). "
              << "Press ] to try starting anyway." << std::endl;
  }
}

bool TeleVuerPlannerInput::ApplyJsonPayload(const std::string& json_body) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(json_body);
  } catch (const std::exception& e) {
    if (zmq_verbose_) std::cerr << "[TeleVuerPlannerInput] JSON parse error: " << e.what() << std::endl;
    return false;
  }

  if (!j.contains("head_pose") || !j.contains("left_wrist_pose") || !j.contains("right_wrist_pose")) {
    if (zmq_verbose_) {
      std::cerr << "[TeleVuerPlannerInput] JSON missing head_pose / left_wrist_pose / right_wrist_pose"
                << std::endl;
    }
    return false;
  }

  double Tl[16], Tr[16], Th[16];
  if (!ParsePoseMatrix4(j["left_wrist_pose"], Tl) || !ParsePoseMatrix4(j["right_wrist_pose"], Tr) ||
      !ParsePoseMatrix4(j["head_pose"], Th)) {
    return false;
  }

  auto tx = [](const double* T) { return T[3]; };
  auto ty = [](const double* T) { return T[7]; };
  auto tz = [](const double* T) { return T[11]; };

  // Freeze head position after first valid frame to keep robot torso/head stable
  // even when the VR headset moves.
  if (!head_position_locked_) {
    locked_head_position_ = {tx(Th), ty(Th), tz(Th)};
    head_position_locked_ = true;
    std::cout << "[TeleVuerPlannerInput] Head position locked at ["
              << locked_head_position_[0] << ", " << locked_head_position_[1] << ", "
              << locked_head_position_[2] << "] (head motion ignored)" << std::endl;
  }

  std::array<double, 9> vr_pos = {tx(Tl), ty(Tl), tz(Tl), tx(Tr), ty(Tr), tz(Tr),
                                  locked_head_position_[0], locked_head_position_[1], locked_head_position_[2]};

  double Rl[9] = {Tl[0], Tl[1], Tl[2], Tl[4], Tl[5], Tl[6], Tl[8], Tl[9], Tl[10]};
  double Rr[9] = {Tr[0], Tr[1], Tr[2], Tr[4], Tr[5], Tr[6], Tr[8], Tr[9], Tr[10]};

  double qlw = 1, qlx = 0, qly = 0, qlz = 0;
  double qrw = 1, qrx = 0, qry = 0, qrz = 0;
  RotmatToQuatWxyz(Rl, qlw, qlx, qly, qlz);
  RotmatToQuatWxyz(Rr, qrw, qrx, qry, qrz);
  const double qhw = 1.0, qhx = 0.0, qhy = 0.0, qhz = 0.0;

  std::array<double, 12> vr_ori = {qlw, qlx, qly, qlz, qrw, qrx, qry, qrz, qhw, qhx, qhy, qhz};

  vr_3point_position_.SetData(vr_pos);
  vr_3point_orientation_.SetData(vr_ori);
  has_vr_3point_control_ = true;

  const auto now = std::chrono::steady_clock::now();
  double dt = 0.005;
  if (have_last_steady_) {
    dt = std::chrono::duration<double>(now - last_steady_for_dt_).count();
    dt = std::max(1e-4, std::min(dt, 0.25));
  }
  have_last_steady_ = true;
  last_steady_for_dt_ = now;

  const bool controller_mode =
      j.contains("mode") && j["mode"].is_string() && j["mode"].get<std::string>() == "controller";

  double jx = 0, jy = 0, rx = 0, ry = 0;
  if (controller_mode) {
    ReadStick2(j, "left_ctrl_thumbstickValue", jx, jy);
    ReadStick2(j, "right_ctrl_thumbstickValue", rx, ry);
    (void)ry;
  }
  rx = -rx;

  PlannerMessage msg;
  msg.valid = true;
  msg.speed = -1.0;
  msg.height = -1.0;

  if (!disable_joystick_locomotion_ && controller_mode) {
    facing_yaw_rad_ += rx * kStickYawScale * dt;
    double c = std::cos(facing_yaw_rad_);
    double s = std::sin(facing_yaw_rad_);
    std::array<double, 3> facing = {c, s, 0.0};

    const double vx_body = -jy;
    const double vy_body = -jx;
    const double mag = std::hypot(vx_body, vy_body);

    if (mag < kStickDeadZone) {
      msg.mode = static_cast<int>(LocomotionMode::IDLE);
      msg.movement = {0.0, 0.0, 0.0};
      msg.facing = facing;
    } else {
      const double mx = vx_body * c - vy_body * s;
      const double my = vx_body * s + vy_body * c;
      std::array<double, 3> movement = {mx, my, 0.0};
      movement = normalize_vector_d(movement);

      const double mag_n =
          std::min(1.0, (mag - kStickDeadZone) / std::max(1e-6, 1.0 - kStickDeadZone));
      if (mag < kWalkStickMag) {
        msg.mode = static_cast<int>(LocomotionMode::SLOW_WALK);
        msg.speed = 0.22 + 0.55 * mag_n;
      } else {
        msg.mode = static_cast<int>(LocomotionMode::WALK);
        msg.speed = -1.0;
      }
      msg.movement = movement;
      msg.facing = facing;
    }
  } else {
    msg.mode = static_cast<int>(LocomotionMode::IDLE);
    msg.movement = {0.0, 0.0, 0.0};
    msg.facing = {std::cos(facing_yaw_rad_), std::sin(facing_yaw_rad_), 0.0};
    msg.facing = normalize_vector_d(msg.facing);
  }

  {
    std::lock_guard<std::mutex> lock(planner_mutex_);
    latest_planner_message_ = msg;
    latest_planner_message_.timestamp = now;
  }
  return true;
}

void TeleVuerPlannerInput::handle_input(MotionDataReader& motion_reader,
                                        std::shared_ptr<const MotionSequence>& current_motion,
                                        int& current_frame,
                                        OperatorState& operator_state,
                                        bool& reinitialize_heading,
                                        DataBuffer<HeadingState>& heading_state_buffer,
                                        bool has_planner,
                                        PlannerState& planner_state,
                                        DataBuffer<MovementState>& movement_state_buffer,
                                        std::mutex& current_motion_mutex,
                                        bool& report_temperature) {
  if (!has_planner) {
    std::cerr << "[TeleVuerPlannerInput] Planner not available" << std::endl;
    operator_state.stop = true;
    return;
  }
  if (report_temperature_flag_) {
    report_temperature = true;
    report_temperature_flag_ = false;
  }
  if (emergency_stop_) {
    operator_state.stop = true;
    if (planner_state.enabled) {
      planner_state.enabled = false;
      planner_state.initialized = false;
    }
    {
      std::lock_guard<std::mutex> lock(planner_mutex_);
      latest_planner_message_.valid = false;
      latest_planner_message_.timestamp = {};
    }
    has_upper_body_control_ = false;
    has_hand_joints_ = false;
    return;
  }

  if (stop_control_) {
    operator_state.stop = true;
    if (planner_state.enabled) {
      planner_state.enabled = false;
      planner_state.initialized = false;
    }
    {
      std::lock_guard<std::mutex> lock(planner_mutex_);
      latest_planner_message_.valid = false;
      latest_planner_message_.timestamp = {};
    }
    has_upper_body_control_ = false;
    has_hand_joints_ = false;
  }

  HandlePlannerModeInput(motion_reader, current_motion, current_frame, operator_state, reinitialize_heading,
                         heading_state_buffer, has_planner, planner_state, movement_state_buffer,
                         current_motion_mutex);
}

void TeleVuerPlannerInput::HandlePlannerModeInput(MotionDataReader& motion_reader,
                                                  std::shared_ptr<const MotionSequence>& current_motion,
                                                  int& current_frame,
                                                  OperatorState& operator_state,
                                                  bool& reinitialize_heading,
                                                  DataBuffer<HeadingState>& heading_state_buffer,
                                                  bool has_planner,
                                                  PlannerState& planner_state,
                                                  DataBuffer<MovementState>& movement_state_buffer,
                                                  std::mutex& current_motion_mutex) {
  (void)motion_reader;
  (void)current_frame;
  (void)heading_state_buffer;
  (void)has_planner;

  if (CheckAndClearSafetyReset()) {
    {
      std::lock_guard<std::mutex> lock(current_motion_mutex);
      operator_state.play = false;
    }
    if (operator_state.start) {
      if (planner_state.enabled && planner_state.initialized) {
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          if (current_motion->GetEncodeMode() == 1) {
            current_motion->SetEncodeMode(0);
          }
          operator_state.play = true;
        }
        std::cout << "[TeleVuerPlannerInput] Safety reset: planner kept enabled" << std::endl;
      } else {
        movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0, 0.0, 0.0},
                                                    {1.0, 0.0, 0.0}, -1.0, -1.0));
        planner_state.enabled = true;
        std::cout << "[TeleVuerPlannerInput] Planner enabled" << std::endl;

        auto wait_start = std::chrono::steady_clock::now();
        constexpr auto kTimeout = std::chrono::seconds(5);
        while (planner_state.enabled) {
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            if (current_motion->name == "planner_motion") break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          if (std::chrono::steady_clock::now() - wait_start > kTimeout) {
            std::cerr << "[TeleVuerPlannerInput] Planner init timeout" << std::endl;
            operator_state.stop = true;
            return;
          }
          std::cout << "[TeleVuerPlannerInput] Waiting for planner_motion..." << std::endl;
        }
        if (!planner_state.enabled || !planner_state.initialized) {
          operator_state.stop = true;
          return;
        }
        is_planner_ready_ = true;
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = true;
        }
      }
    }
    return;
  }

  if (start_control_ && !operator_state.start) {
    operator_state.start = true;
    {
      std::lock_guard<std::mutex> lock(current_motion_mutex);
      operator_state.play = false;
      reinitialize_heading = true;
    }
    if (!planner_state.enabled) {
      planner_state.enabled = true;
      std::cout << "[TeleVuerPlannerInput] Planner enabled" << std::endl;
    }
    auto wait_start = std::chrono::steady_clock::now();
    constexpr auto kTimeout = std::chrono::seconds(5);
    while (planner_state.enabled) {
      {
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        if (current_motion->name == "planner_motion") break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (std::chrono::steady_clock::now() - wait_start > kTimeout) {
        std::cerr << "[TeleVuerPlannerInput] Planner init timeout" << std::endl;
        operator_state.stop = true;
        return;
      }
      std::cout << "[TeleVuerPlannerInput] Waiting for planner_motion..." << std::endl;
    }
    if (!planner_state.enabled || !planner_state.initialized) {
      operator_state.stop = true;
      return;
    }
    is_planner_ready_ = true;
    {
      std::lock_guard<std::mutex> lock(current_motion_mutex);
      operator_state.play = true;
    }
  }

  if (planner_state.enabled && planner_state.initialized) {
    std::lock_guard<std::mutex> lock(planner_mutex_);

    constexpr auto kPlannerTimeout = std::chrono::milliseconds(1000);
    auto time_since = std::chrono::steady_clock::now() - latest_planner_message_.timestamp;

    if (latest_planner_message_.valid) {
      has_upper_body_control_ = latest_planner_message_.upper_body_position.has_value();
      has_hand_joints_ = latest_planner_message_.left_hand_joints.has_value() ||
                         latest_planner_message_.right_hand_joints.has_value();

      MovementState mode_state(latest_planner_message_.mode, latest_planner_message_.movement,
                               latest_planner_message_.facing, latest_planner_message_.speed,
                               latest_planner_message_.height);

      if (is_squat_motion_mode(static_cast<LocomotionMode>(mode_state.locomotion_mode))) {
        if (mode_state.height < 0.2) mode_state.height = 0.2;
      }
      if (is_static_motion_mode(static_cast<LocomotionMode>(mode_state.locomotion_mode))) {
        mode_state.movement_speed = -1.0;
      }

      mode_state.facing_direction = normalize_vector_d(mode_state.facing_direction);
      mode_state.movement_direction = normalize_vector_d(mode_state.movement_direction);

      movement_state_buffer.SetData(mode_state);
      latest_planner_message_.valid = false;
    } else if (!latest_planner_message_.valid && time_since >= kPlannerTimeout) {
      has_upper_body_control_ = false;
      has_hand_joints_ = false;
      auto current_facing = movement_state_buffer.GetDataWithTime().data->facing_direction;
      MovementState idle_state(static_cast<int>(LocomotionMode::IDLE), {0.0, 0.0, 0.0}, current_facing, -1.0,
                               -1.0);
      movement_state_buffer.SetData(idle_state);

      if (latest_planner_message_.timestamp != std::chrono::steady_clock::time_point{}) {
        std::cout << "[TeleVuerPlannerInput] Planner timeout → IDLE" << std::endl;
        latest_planner_message_.valid = false;
        latest_planner_message_.timestamp = {};
      }
    }
  }

  if (has_vr_3point_control_ && !last_has_vr_3point_control_) {
    std::cout << "[TeleVuerPlannerInput] VR 3-point control enabled" << std::endl;
    std::lock_guard<std::mutex> lock(current_motion_mutex);
    if (current_motion->GetEncodeMode() >= 0) {
      current_motion->SetEncodeMode(1);
    }
  } else if (!has_vr_3point_control_ && last_has_vr_3point_control_) {
    std::cout << "[TeleVuerPlannerInput] VR 3-point control disabled" << std::endl;
    std::lock_guard<std::mutex> lock(current_motion_mutex);
    if (current_motion->GetEncodeMode() >= 0) {
      current_motion->SetEncodeMode(0);
    }
  }
  last_has_vr_3point_control_ = has_vr_3point_control_;
}
