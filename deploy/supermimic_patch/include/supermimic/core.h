#pragma once

// SuperMimic deployment core: observation layout, action mapping and rate/limit guards.
//
// Deliberately free of DDS and inference: the same code drives the real controller and
// the offline MuJoCo verification executable, and it is what
// ``supermimic/scripts/replay_headless.py --deploy-onnx`` mirrors in Python.
//
// Observation (68 floats, "rep_state" of GenCoordRepresentation):
//   [z(1), g_b(3), v_b(3), w_b(3), qj(29), vj(29)]
// Goal comes first in the ONNX input vector, so the model input is
//   obs = [goal(goal_dim), rep_state(68)]
// and for the velocity task goal_dim = 3 -> 71.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "beyondmimic/core.h"  // Snapshot / Config / LegOdometry / forward_kinematics

namespace supermimic {

constexpr int J = beyondmimic::J;          // 29 joints
constexpr int REP_STATE_DIM = 68;          // z + 3 + 3 + 3 + 2 * J
constexpr int VELOCITY_GOAL_DIM = 3;       // [vx_b, vy_b, wz_b]
constexpr int VELOCITY_OBS_DIM = VELOCITY_GOAL_DIM + REP_STATE_DIM;

static_assert(REP_STATE_DIM == 10 + 2 * J, "rep_state layout must be 10 + 2J");

using Vec = Eigen::Vector3f;
using Quat = Eigen::Quaternionf;
using Joints = std::array<float, J>;
using VelocityGoal = std::array<float, VELOCITY_GOAL_DIM>;

struct VelocityClip {
  float vx_lo = -0.1f, vx_hi = 1.0f;
  float vy_lo = -0.1f, vy_hi = 0.1f;
  float wz_lo = -1.5f, wz_hi = 1.5f;

  VelocityGoal clamp(const VelocityGoal& goal) const {
    return {
      std::clamp(goal[0], vx_lo, vx_hi),
      std::clamp(goal[1], vy_lo, vy_hi),
      std::clamp(goal[2], wz_lo, wz_hi),
    };
  }
  void validate() const {
    if (!(vx_lo < vx_hi && vy_lo < vy_hi && wz_lo < wz_hi)) {
      throw std::runtime_error("Invalid velocity command clip");
    }
  }
};

struct SafetyLimits {
  float max_tilt_rad = 0.85f;        // abort when the pelvis tilts past this
  float max_target_speed = 8.0f;     // rad/s, rate limit on the position target
  float max_anchor_error = 0.35f;    // unused by velocity mode, kept for motion mode
  void validate() const {
    if (!(max_tilt_rad > 0 && max_tilt_rad < 1.5f)) {
      throw std::runtime_error("Invalid tilt limit");
    }
    if (!(max_target_speed > 0 && max_target_speed <= 20.0f)) {
      throw std::runtime_error("Invalid target rate limit");
    }
  }
};

// Rotate a world vector into the body frame (inverse rotation, wxyz quaternion).
inline Vec rotate_inverse(const Quat& q, const Vec& v) {
  return q.conjugate() * v;
}

inline float tilt_angle(const Quat& q) {
  // Angle between the body z-axis and world z.
  const Vec body_z = q * Vec::UnitZ();
  return std::acos(std::clamp(body_z.z(), -1.0f, 1.0f));
}

class Core {
public:
  Core(beyondmimic::Config robot, VelocityClip clip, SafetyLimits limits)
      : robot_(std::move(robot)), clip_(clip), limits_(limits), odometry_(robot_) {
    robot_.validate();
    clip_.validate();
    limits_.validate();
  }

  // ``initial_height`` is the pelvis height above the floor at the moment the policy
  // starts; the state machine derives it from the foot kinematics (see estimate_height).
  void reset(const beyondmimic::Snapshot& snapshot, float initial_height) {
    beyondmimic::validate_snapshot(snapshot);
    odometry_.reset(snapshot, Vec(0.0f, 0.0f, initial_height));
    last_target_ = snapshot.q;
    last_action_.fill(0.0f);
    clipped_ = 0;
  }

  // Builds the exact model input: [goal, rep_state].
  std::array<float, VELOCITY_OBS_DIM> observe(
      const beyondmimic::Snapshot& snapshot,
      const VelocityGoal& raw_goal,
      bool update_odometry = true,
      const Vec* measured_position = nullptr,
      const Vec* measured_velocity_b = nullptr) {
    beyondmimic::validate_snapshot(snapshot);
    if (update_odometry) {
      odometry_.update(snapshot);
    }
    const Vec position = measured_position ? *measured_position : odometry_.position();
    const Vec velocity_b = measured_velocity_b ? *measured_velocity_b : odometry_.velocity_b();
    const VelocityGoal goal = clip_.clamp(raw_goal);

    std::array<float, VELOCITY_OBS_DIM> obs{};
    obs[0] = goal[0];
    obs[1] = goal[1];
    obs[2] = goal[2];

    float* rep = obs.data() + VELOCITY_GOAL_DIM;
    rep[0] = position.z();
    const Vec gravity_body = rotate_inverse(snapshot.orientation, Vec(0.0f, 0.0f, -1.0f));
    rep[1] = gravity_body.x();
    rep[2] = gravity_body.y();
    rep[3] = gravity_body.z();
    rep[4] = velocity_b.x();
    rep[5] = velocity_b.y();
    rep[6] = velocity_b.z();
    rep[7] = snapshot.angular_velocity.x();
    rep[8] = snapshot.angular_velocity.y();
    rep[9] = snapshot.angular_velocity.z();
    for (int i = 0; i < J; ++i) {
      rep[10 + i] = snapshot.q[i];        // raw joint angle: gencoord does NOT subtract default
      rep[10 + J + i] = snapshot.dq[i];
    }
    return obs;
  }

  // q_des = default + action_scale * action, then limit, rate-limit and count clipping.
  Joints targets(const beyondmimic::Snapshot& snapshot, const Joints& action) {
    Joints result{};
    const float max_step = limits_.max_target_speed * robot_.dt;
    for (int i = 0; i < J; ++i) {
      float desired = robot_.default_q[i] + robot_.action_scale[i] * action[i];
      desired = std::clamp(desired, robot_.lower[i] + 0.01f, robot_.upper[i] - 0.01f);
      desired = std::clamp(desired, last_target_[i] - max_step, last_target_[i] + max_step);
      if (std::abs(desired - (robot_.default_q[i] + robot_.action_scale[i] * action[i])) >
          1e-5f) {
        ++clipped_;
      }
      result[i] = desired;
      last_target_[i] = desired;
    }
    last_action_ = action;
    return result;
  }

  // True when the robot is upright enough to keep running.
  bool healthy(const beyondmimic::Snapshot& snapshot) const {
    return tilt_angle(snapshot.orientation) <= limits_.max_tilt_rad;
  }

  int clipped() const { return clipped_; }
  const beyondmimic::Config& config() const { return robot_; }
  const VelocityClip& clip() const { return clip_; }

  // Pelvis height implied by the current joint angles with both soles on the floor.
  float estimate_height(const beyondmimic::Snapshot& snapshot) const {
    const auto left = beyondmimic::forward_kinematics(robot_.left_foot, snapshot.q);
    const auto right = beyondmimic::forward_kinematics(robot_.right_foot, snapshot.q);
    const float lowest = std::min(left.translation().z(), right.translation().z());
    return -lowest;
  }

private:
  beyondmimic::Config robot_;
  VelocityClip clip_;
  SafetyLimits limits_;
  beyondmimic::LegOdometry odometry_;
  Joints last_target_{};
  Joints last_action_{};
  int clipped_ = 0;
};

}  // namespace supermimic