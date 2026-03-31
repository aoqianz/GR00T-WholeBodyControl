/**
 * @file televuer_planner_input.hpp
 * @brief Direct TeleVuer ZMQ IPC JSON → planner + VR 3-point (no Python bridge).
 *
 * Subscribes to the same multipart stream as teleop/televuer (topic + JSON body):
 *   head_pose, left_wrist_pose, right_wrist_pose with 4x4 "matrix".
 * Optional controller thumbsticks map to locomotion (matches televr_sonic_zmq_bridge.py).
 * VR 3-point: wrists from TeleVuer poses; head XZ locked after first head_pose; head Y = locked Y +
 *   k * (left_squeeze - right_squeeze) from televuer/zmq_wrapper.py (ctrl: *_ctrl_squeezeValue,
 *   hand: *_hand_squeezeValue). Head orientation fixed (identity quat).
 *
 * Wire format: televuer/zmq_wrapper.py TeleDataZmqPublisher._build_payload
 */

#ifndef TELEVUER_PLANNER_INPUT_HPP
#define TELEVUER_PLANNER_INPUT_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <termios.h>

#include "input_interface.hpp"
#include "input_command.hpp"

struct zmq_ctx_t;
struct zmq_socket_t;

/**
 * @class TeleVuerPlannerInput
 * @brief PLANNER-mode input from TeleVuer IPC JSON only (no TCP command/planner topics).
 */
class TeleVuerPlannerInput : public InputInterface {
 public:
  TeleVuerPlannerInput(
      std::string ipc_path,
      std::string topic = "televuer.teledata",
      bool disable_joystick_locomotion = false,
      bool zmq_verbose = false);

  ~TeleVuerPlannerInput() override;

  void update() override;

  void handle_input(MotionDataReader& motion_reader,
                    std::shared_ptr<const MotionSequence>& current_motion,
                    int& current_frame,
                    OperatorState& operator_state,
                    bool& reinitialize_heading,
                    DataBuffer<HeadingState>& heading_state_buffer,
                    bool has_planner,
                    PlannerState& planner_state,
                    DataBuffer<MovementState>& movement_state_buffer,
                    std::mutex& current_motion_mutex,
                    bool& report_temperature) override;

  InputType GetType() override { return InputType::NETWORK; }

  bool HasVR3PointControl() const override { return has_vr_3point_control_; }

 private:
  void DrainSocketAndApplyLatest();
  bool ApplyJsonPayload(const std::string& json_body);
  void PublishVR3PointFromCaches();
  void HandlePlannerModeInput(MotionDataReader& motion_reader,
                              std::shared_ptr<const MotionSequence>& current_motion,
                              int& current_frame,
                              OperatorState& operator_state,
                              bool& reinitialize_heading,
                              DataBuffer<HeadingState>& heading_state_buffer,
                              bool has_planner,
                              PlannerState& planner_state,
                              DataBuffer<MovementState>& movement_state_buffer,
                              std::mutex& current_motion_mutex);

  std::string ipc_path_;
  std::string topic_;
  bool disable_joystick_locomotion_;
  bool zmq_verbose_;
  std::string config_path_ = "policy/release/televuer_config.yaml";

  void* zmq_ctx_ = nullptr;
  void* zmq_socket_ = nullptr;

  std::mutex planner_mutex_;
  PlannerMessage latest_planner_message_;

  double facing_yaw_rad_ = 0.0;
  std::chrono::steady_clock::time_point last_steady_for_dt_{};
  bool have_last_steady_ = false;

  bool emergency_stop_ = false;
  bool report_temperature_flag_ = false;
  bool start_control_ = false;
  bool stop_control_ = false;

  bool is_planner_ready_ = false;
  bool last_has_vr_3point_control_ = false;

  /// Increments when a full update cycle had no successful JSON apply; used for rate-limited hints.
  int no_teledata_log_counter_ = 0;

  /// Third VR point position: fixed after first valid head_pose (ignores later headset translation).
  bool head_position_locked_ = false;
  std::array<double, 3> locked_head_position_{0.0, 0.0, 0.0};
  std::array<double, 6> cached_wrist_xyz_{};
  std::array<double, 8> cached_wrist_q_wxyz_{};  // left wxyz + right wxyz
  bool vr_wrist_cache_valid_ = false;

  /// Latest TeleVuer grip/squeeze analog [0,1] per zmq_wrapper _build_payload (OpenXR squeeze = grip).
  double last_left_squeeze_ = 0.0;
  double last_right_squeeze_ = 0.0;
  /// False: VR live control; True: track mode with YAML constants.
  bool tracking_mode_ = false;
  /// Track-mode constant planar velocity [vx, vy] in robot/body frame.
  std::array<double, 2> tracking_velocity_xy_{0.0, 0.0};
  /// Track-mode constant left/right wrist XYZ for policy 3-point input.
  std::array<double, 3> tracking_left_wrist_xyz_{0.25, 0.15, 0.25};
  std::array<double, 3> tracking_right_wrist_xyz_{0.25, -0.15, 0.25};
  /// Track-mode constant left/right wrist quaternions (wxyz).
  std::array<double, 4> tracking_left_wrist_q_wxyz_{1.0, 0.0, 0.0, 0.0};
  std::array<double, 4> tracking_right_wrist_q_wxyz_{1.0, 0.0, 0.0, 0.0};

  /// When stdin is a TTY, saved attributes to restore in dtor (see SimpleKeyboard).
  struct termios stdin_saved_termios_{};
  bool stdin_tty_raw_mode_ = false;

  // Joystick tuning (aligned with televr_sonic_zmq_bridge.py defaults)
  static constexpr double kStickDeadZone = 0.12;
  static constexpr double kStickYawScale = 1.8;
  static constexpr double kWalkStickMag = 0.72;
  /// Default joystick translational velocity gain if config is absent/invalid.
  static constexpr double kDefaultVelocityGain = 0.8;
  /// Runtime velocity gain loaded from YAML config.
  double velocity_gain_ = kDefaultVelocityGain;
  /// Max head Y shift (meters) when one grip is fully squeezed and the other is open (difference ∈ [-1,1]).
  static constexpr double kHeadGripYScaleM = 0.35;
};

#endif  // TELEVUER_PLANNER_INPUT_HPP
