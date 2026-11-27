#include "State_SuperMimic.h"

#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace {
constexpr float kPolicyHz = 50.0f;          // package step_dt = 0.02 s
constexpr int kOutputTimeoutMs = 100;       // same watchdog as the BeyondMimic state
constexpr float kCommandReloadS = 0.1f;     // velocity command file poll period
}  // namespace

State_SuperMimic::State_SuperMimic(int state, std::string name) : FSMState(state, name) {
  auto cfg = param::config["FSM"][name];
  auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
  package_ = std::make_unique<supermimic::Package>(
      supermimic::load_package(policy_dir / "deploy.yaml"));
  core_ = std::make_unique<supermimic::Core>(package_->robot, package_->clip,
                                             package_->limits);
  runner_ = std::make_unique<supermimic::Runner>(package_->policy,
                                                 supermimic::VELOCITY_OBS_DIM);
  lowstate->set_timeout_ms(kOutputTimeoutMs);

  command_file_ = (param::proj_dir / cfg["command_file"].as<std::string>(
                                        "config/velocity_cmd.yaml"))
                      .string();

  registered_checks.insert(registered_checks.begin(),
                           {[this] { return stopped_.load(); },
                            FSMStringMap.right.at("Passive")});

  const auto& clip = package_->clip;
  spdlog::info(
      "SuperMimic loaded: obs {} (goal 3 + rep 68), 29 actions, clip vx[{:.2f},{:.2f}] "
      "vy[{:.2f},{:.2f}] wz[{:.2f},{:.2f}], command file {}",
      supermimic::VELOCITY_OBS_DIM, clip.vx_lo, clip.vx_hi, clip.vy_lo, clip.vy_hi,
      clip.wz_lo, clip.wz_hi, command_file_);
}

beyondmimic::Snapshot State_SuperMimic::snapshot() {
  std::lock_guard<std::mutex> lock(lowstate->mutex_);
  beyondmimic::Snapshot result;
  const auto& msg = lowstate->msg_;
  for (int i = 0; i < beyondmimic::J; ++i) {
    result.q[i] = msg.motor_state()[i].q();
    result.dq[i] = msg.motor_state()[i].dq();
  }
  const auto& quat = msg.imu_state().quaternion();
  const auto& gyro = msg.imu_state().gyroscope();
  result.orientation = beyondmimic::Quat(quat[0], quat[1], quat[2], quat[3]);
  result.angular_velocity = beyondmimic::Vec(gyro[0], gyro[1], gyro[2]);
  beyondmimic::validate_snapshot(result);
  result.orientation.normalize();
  return result;
}

void State_SuperMimic::apply_gains() {
  const auto& cfg = core_->config();
  for (int i = 0; i < beyondmimic::J; ++i) {
    auto& motor = lowcmd->msg_.motor_cmd()[cfg.sdk_indices[i]];
    motor.kp() = cfg.kp[i];
    motor.kd() = cfg.kd[i];
  }
}

// Velocity command source: a small YAML file so the operator (or a test harness) can
// change the goal without rebuilding. Replace this with the analog stick reading
// (``lowstate->joystick.lx/ly/rx``) once the SDK's analog accessor is confirmed; every
// command still goes through the package's velocity clip below.
supermimic::VelocityGoal State_SuperMimic::read_command() {
  const auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration<float>(now - last_command_read_).count() < kCommandReloadS) {
    return command_;
  }
  last_command_read_ = now;
  try {
    const auto yaml = YAML::LoadFile(command_file_);
    command_ = {
        yaml["vx"].as<float>(0.0f),
        yaml["vy"].as<float>(0.0f),
        yaml["wz"].as<float>(0.0f),
    };
  } catch (const std::exception& e) {
    // Missing/edited file: hold the last command rather than dropping to zero mid-stride.
    spdlog::warn("Velocity command file unreadable ({}), holding {:.2f},{:.2f},{:.2f}",
                 e.what(), command_[0], command_[1], command_[2]);
  }
  return command_;
}

void State_SuperMimic::enter() {
  exit();
  stopped_ = false;
  try {
    if (lowstate->isTimeout()) {
      throw std::runtime_error("LowState timeout at policy start");
    }
    const auto current = snapshot();
    if (supermimic::tilt_angle(current.orientation) > package_->limits.max_tilt_rad) {
      throw std::runtime_error("Robot is tilted too far to start SuperMimic");
    }
    core_->reset(current, core_->estimate_height(current));
    command_ = {0.0f, 0.0f, 0.0f};
    last_command_read_ = std::chrono::steady_clock::now();
    targets_ = current.q;  // hold the entry pose for the first tick
    apply_gains();
    running_ = true;
    worker_ = std::thread(&State_SuperMimic::policy_loop, this);
  } catch (const std::exception& e) {
    stop(e.what());
  }
}

void State_SuperMimic::policy_loop() {
  const auto period = std::chrono::duration<double>(1.0 / kPolicyHz);
  auto next = std::chrono::steady_clock::now() + period;
  try {
    while (running_) {
      const auto current = snapshot();
      if (!core_->healthy(current)) {
        stop("Tilt limit exceeded");
        break;
      }
      const auto goal = read_command();
      const auto obs = core_->observe(current, goal);
      const auto action = runner_->act(obs);
      const auto targets = core_->targets(current, action);
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        targets_ = targets;
        last_result_ = std::chrono::steady_clock::now();
      }
      std::this_thread::sleep_until(next);
      next += period;
    }
  } catch (const std::exception& e) {
    stop(e.what());
  }
}

void State_SuperMimic::run() {
  if (stopped_) {
    damping();
    return;
  }
  std::lock_guard<std::mutex> lock(target_mutex_);
  if (std::chrono::steady_clock::now() - last_result_ >
      std::chrono::milliseconds(kOutputTimeoutMs)) {
    stop("Policy output timeout");
    damping();
    return;
  }
  const auto& cfg = core_->config();
  for (int i = 0; i < beyondmimic::J; ++i) {
    lowcmd->msg_.motor_cmd()[cfg.sdk_indices[i]].q() = targets_[i];
  }
}

void State_SuperMimic::damping() {
  for (int i = 0; i < beyondmimic::J; ++i) {
    auto& motor = lowcmd->msg_.motor_cmd()[i];
    motor.kp() = 0;
    motor.kd() = 3;
    motor.dq() = 0;
    motor.tau() = 0;
  }
}

void State_SuperMimic::stop(const std::string& reason) {
  if (!stopped_.exchange(true)) {
    spdlog::warn("SuperMimic stopping: {} (clipped {} targets)", reason, core_->clipped());
  }
  running_ = false;
}

void State_SuperMimic::exit() {
  running_ = false;
  if (worker_.joinable()) {
    worker_.join();
  }
}