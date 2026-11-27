#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include "FSM/FSMState.h"
#include "supermimic/package.h"

// SuperMimic velocity-command state.
//
// Reuses the BeyondMimic state-machine skeleton (20 ms policy thread, 1 kHz publishing,
// 100 ms output watchdog, damping on abort) but drives the SuperMimic actor: the
// observation is the deploy ``rep_state`` + a 3-D velocity goal, and the action is mapped
// back to joint targets with ``default + action_scale * a``.
class State_SuperMimic : public FSMState {
 public:
  State_SuperMimic(int state, std::string name);
  ~State_SuperMimic() override { exit(); }

  void enter() override;
  void run() override;
  void exit() override;

 private:
  beyondmimic::Snapshot snapshot();
  void policy_loop();
  void stop(const std::string& reason);
  void damping();
  void apply_gains();
  supermimic::VelocityGoal read_command();

  std::unique_ptr<supermimic::Package> package_;
  std::unique_ptr<supermimic::Core> core_;
  std::unique_ptr<supermimic::Runner> runner_;

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopped_{true};
  std::mutex target_mutex_;
  beyondmimic::Joints targets_{};
  std::chrono::steady_clock::time_point last_result_{};

  std::string command_file_;
  supermimic::VelocityGoal command_{};
  std::chrono::steady_clock::time_point last_command_read_{};
};

REGISTER_FSM(State_SuperMimic)