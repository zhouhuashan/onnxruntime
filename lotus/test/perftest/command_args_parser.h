#pragma once

namespace Lotus {
namespace PerfTest {

struct PerformanceTestConfig;

class CommandLineParser {
 public:
  static void ShowUsage();

  static bool ParseArguments(PerformanceTestConfig& test_config, int argc, char* argv[]);
};

}  // namespace PerfTest
}  // namespace Lotus