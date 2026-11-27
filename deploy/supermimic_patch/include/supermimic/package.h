#pragma once

// Package loading + ONNX inference for SuperMimic.
//
// The ONNX contract is intentionally identical to the BeyondMimic package so the same
// ``Runner`` validation code applies:
//   input  ``obs``     float32 [1, goal_dim + 68]
//   output ``actions`` float32 [1, 29]

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include "beyondmimic/core.h"
#include "supermimic/core.h"

namespace supermimic {

template <std::size_t N>
std::array<float, N> read_array(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() != N) {
    throw std::runtime_error("Invalid package vector length");
  }
  std::array<float, N> result{};
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = node[i].as<float>();
    if (!std::isfinite(result[i])) {
      throw std::runtime_error("Non-finite package parameter");
    }
  }
  return result;
}

struct Package {
  beyondmimic::Config robot;
  VelocityClip clip;
  SafetyLimits limits;
  Joints entry_pose{};        // where FixStand should leave the robot
  float max_start_error = 0.6f;  // rad, per-joint deviation accepted at takeover
  int goal_dim = VELOCITY_GOAL_DIM;
  std::filesystem::path policy;
};

inline beyondmimic::Chain read_chain(const YAML::Node& node) {
  beyondmimic::Chain chain;
  for (const auto& n : node) {
    auto xyz = read_array<3>(n["xyz"]);
    auto rpy = read_array<3>(n["rpy"]);
    auto axis = read_array<3>(n["axis"]);
    chain.push_back({n["motor_index"].as<int>(), beyondmimic::Vec(xyz.data()),
                     beyondmimic::Vec(rpy.data()), beyondmimic::Vec(axis.data())});
  }
  return chain;
}

inline Package load_package(const std::filesystem::path& deploy_yaml) {
  const auto yaml = YAML::LoadFile(deploy_yaml.string());
  if (yaml["format"].as<std::string>() != "supermimic_g1_gencoord_v1") {
    throw std::runtime_error("Not a SuperMimic deploy package");
  }
  if (yaml["rep_state_dim"].as<int>() != REP_STATE_DIM ||
      yaml["action_dim"].as<int>() != J ||
      !yaml["normalizer"]["embedded"].as<bool>()) {
    throw std::runtime_error("Incompatible SuperMimic package dimensions");
  }

  Package pkg;
  pkg.robot.dt = yaml["step_dt"].as<float>();
  pkg.robot.default_q = read_array<J>(yaml["default_joint_pos"]);
  pkg.robot.kp = read_array<J>(yaml["kp"]);
  pkg.robot.kd = read_array<J>(yaml["kd"]);
  pkg.robot.effort = read_array<J>(yaml["effort_limit"]);
  pkg.robot.lower = read_array<J>(yaml["joint_lower"]);
  pkg.robot.upper = read_array<J>(yaml["joint_upper"]);

  // action_scale may be a scalar (all joints) or a per-joint sequence.
  const auto scale_node = yaml["action_scale"];
  if (scale_node.IsScalar()) {
    const float scalar = scale_node.as<float>();
    for (int i = 0; i < J; ++i) {
      pkg.robot.action_scale[i] = scalar;
    }
  } else {
    pkg.robot.action_scale = read_array<J>(scale_node);
  }

  if (yaml["sdk_indices"].size() != J) {
    throw std::runtime_error("Invalid SDK mapping");
  }
  for (int i = 0; i < J; ++i) {
    pkg.robot.sdk_indices[i] = yaml["sdk_indices"][i].as<int>();
  }

  pkg.robot.torso = read_chain(yaml["kinematics"]["torso"]);
  pkg.robot.left_foot = read_chain(yaml["kinematics"]["left_foot"]);
  pkg.robot.right_foot = read_chain(yaml["kinematics"]["right_foot"]);
  pkg.robot.validate();

  const auto safety = yaml["safety"];
  pkg.clip.vx_lo = safety["velocity_command_clip"]["vx"][0].as<float>();
  pkg.clip.vx_hi = safety["velocity_command_clip"]["vx"][1].as<float>();
  pkg.clip.vy_lo = safety["velocity_command_clip"]["vy"][0].as<float>();
  pkg.clip.vy_hi = safety["velocity_command_clip"]["vy"][1].as<float>();
  pkg.clip.wz_lo = safety["velocity_command_clip"]["wz"][0].as<float>();
  pkg.clip.wz_hi = safety["velocity_command_clip"]["wz"][1].as<float>();
  pkg.clip.validate();
  pkg.entry_pose = read_array<J>(yaml["entry_pose"]);
  pkg.max_start_error = safety["max_start_error"].as<float>();
  pkg.limits.max_tilt_rad = safety["max_tilt_rad"].as<float>();
  pkg.limits.max_target_speed = safety["max_target_speed_rad_s"].as<float>();
  pkg.limits.validate();

  // The package may expose several task graphs; the velocity slot is the one that needs
  // no reference motion, and it is what the controller runs by default.
  const auto policy = yaml["policy"];
  pkg.policy = deploy_yaml.parent_path() /
               (policy["velocity"] ? policy["velocity"].as<std::string>()
                                  : policy.begin()->second.as<std::string>());
  pkg.goal_dim = yaml["goal_dim"]["velocity"].as<int>();
  if (pkg.goal_dim != VELOCITY_GOAL_DIM) {
    throw std::runtime_error("Velocity goal must be 3-dimensional");
  }
  if (!std::filesystem::exists(pkg.policy)) {
    throw std::runtime_error("Missing policy graph: " + pkg.policy.string());
  }
  return pkg;
}

class Runner {
public:
  Runner(const std::filesystem::path& path, int obs_dim)
      : env_(ORT_LOGGING_LEVEL_WARNING, "supermimic") {
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options_);
    Ort::AllocatorWithDefaultOptions allocator;
    const auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1 ||
        std::string(session_->GetInputNameAllocated(0, allocator).get()) != "obs" ||
        std::string(session_->GetOutputNameAllocated(0, allocator).get()) != "actions" ||
        shape != std::vector<int64_t>{1, obs_dim} ||
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error(
          "Model must be a SuperMimic [1,obs_dim] -> [1,29] float32 ONNX");
    }
    obs_dim_ = obs_dim;
  }

  template <std::size_t N>
  Joints act(const std::array<float, N>& obs) {
    static_assert(N == VELOCITY_OBS_DIM, "velocity policy expects the 71-float input");
    const std::array<int64_t, 2> shape{1, N};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(obs.data()), N,
                                                 shape.data(), shape.size());
    const char* inputs[]{"obs"};
    const char* outputs[]{"actions"};
    auto result = session_->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
    Joints action{};
    std::copy_n(result[0].GetTensorData<float>(), J, action.begin());
    return action;
  }

  int obs_dim() const { return obs_dim_; }

private:
  Ort::Env env_;
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  int obs_dim_ = 0;
};

}  // namespace supermimic