#pragma once

#include <cstdint>
#include <string>

#include "core/graph/constants.h"

namespace Lotus {
namespace PerfTest {

enum class TestMode : std::uint8_t {
  kFixDurationMode = 0,
  KFixRepeatedTimesMode
};

enum class Platform : std::uint8_t {
  kWindows = 0,
  kLinux
};

struct ModelInfo{
  std::string model_name;
  std::string model_file_path;
  std::string input_file_path;
  std::string result_file_path;
};

struct MachineConfig{
  Platform platform{Platform::kWindows};
  std::string provider_type_name{LotusIR::kCpuExecutionProvider};
};

struct RunConfig{
  std::string profile_file;
  TestMode test_mode{TestMode::kFixDurationMode};
  size_t repeated_times{1000};
  size_t duration_in_seconds{600};
  bool f_verbose{false};
};

struct PerformanceTestConfig {
  ModelInfo model_info;
  MachineConfig machine_config;
  RunConfig run_config;

};

}  // PerfTest
}  // Lotus