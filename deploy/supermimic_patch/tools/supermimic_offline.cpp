// Hardware-free parity check for the SuperMimic deploy path.
//
// Feeds recorded `beyondmimic::Snapshot` rows through ``supermimic::Core::observe`` +
// ``Runner`` and prints the resulting joint targets. Record the same rows from the
// simulator (or from an mjlab rollout) with the Python harness and diff the two outputs:
// this is the level at which a swapped joint index or a missed normalizer shows up.
//
// Input CSV, one row per 20 ms policy step:
//   q[29], dq[29], qw, qx, qy, qz, gx, gy, gz, goal_vx, goal_vy, goal_wz
//   (q/dq in policy order, quaternion wxyz, gyro in the BODY frame; 68 columns)

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "supermimic/package.h"

namespace {

std::vector<float> parse_row(const std::string& line) {
  std::vector<float> values;
  std::stringstream stream(line);
  std::string cell;
  while (std::getline(stream, cell, ',')) {
    if (cell.empty()) {
      continue;
    }
    values.push_back(std::stof(cell));
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: supermimic_offline <package-dir> <snapshots.csv>\n";
    return 2;
  }
  try {
    const std::filesystem::path dir = argv[1];
    const auto package = supermimic::load_package(dir / "deploy.yaml");
    supermimic::Core core(package.robot, package.clip, package.limits);
    supermimic::Runner runner(package.policy, supermimic::VELOCITY_OBS_DIM);

    std::ifstream file(argv[2]);
    if (!file) {
      throw std::runtime_error("cannot open snapshots: " + std::string(argv[2]));
    }
    std::string line;
    std::size_t rows = 0;
    bool initialised = false;
    std::printf("step,action0,action1,target0,target1\n");
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      const auto values = parse_row(line);
      if (values.size() != 68) {
        throw std::runtime_error("expected 68 columns per row, got " +
                                 std::to_string(values.size()));
      }
      beyondmimic::Snapshot snapshot;
      for (int i = 0; i < beyondmimic::J; ++i) {
        snapshot.q[i] = values[i];
        snapshot.dq[i] = values[beyondmimic::J + i];
      }
      snapshot.orientation = beyondmimic::Quat(values[58], values[59], values[60],
                                              values[61]);
      snapshot.orientation.normalize();
      snapshot.angular_velocity = beyondmimic::Vec(values[62], values[63], values[64]);
      const supermimic::VelocityGoal goal{values[65], values[66], values[67]};

      if (!initialised) {
        core.reset(snapshot, core.estimate_height(snapshot));
        initialised = true;
      }
      const auto obs = core.observe(snapshot, goal);
      const auto action = runner.act(obs);
      const auto targets = core.targets(snapshot, action);
      std::printf("%zu,%.5f,%.5f,%.5f,%.5f\n", rows, action[0], action[1], targets[0],
                  targets[1]);
      ++rows;
    }
    std::cerr << "processed " << rows << " steps, clipped " << core.clipped() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[FATAL] " << e.what() << "\n";
    return 1;
  }
  return 0;
}