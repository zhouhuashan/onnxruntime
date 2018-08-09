#pragma once

#include <fstream>
#include <string>
#include <vector>

// Lotus dependencies
#include <core/common/logging/sinks/clog_sink.h>
#include <core/common/logging/logging.h>
#include <core/framework/environment.h>
#include <core/session/inference_session.h>
#include <core/platform/env.h>
#include <core/session/IOBinding.h>

#include "test_configuration.h"

namespace Lotus {
namespace PerfTest {

struct PerformanceResult {
  size_t peak_workingset_size{0};
  short average_CPU_usage{0};
  double total_time_cost{0};
  std::vector<double> time_costs;
  std::string model_name;

  void DumpToFile(const std::string& path) const {
    std::ofstream outfile;
    outfile.open(path, std::ofstream::out | std::ofstream::app);
    if (!outfile.good()) {
      LOGF_DEFAULT(ERROR, "failed to open result file");
      return;
    }
    for (size_t runs = 0; runs < time_costs.size(); runs++) {
      outfile << model_name << "," << time_costs[runs] << "," << peak_workingset_size << "," << average_CPU_usage << "," << runs << std::endl;
    }
    outfile.close();
  }
};

class PerformanceRunner {
 public:
  PerformanceRunner(const PerformanceTestConfig& test_config) : performance_test_config_(test_config) {}

  void Run();

  inline const PerformanceResult& GetResult() const { return performance_result_; }

  inline void SerializeResult() const { performance_result_.DumpToFile(performance_test_config_.model_info.result_file_path); }

 private:
  bool Initialize();

  inline void RunOneIteration(bool isWarmup = false) {
    auto start = std::chrono::high_resolution_clock::now();
    session_object_->Run(*io_binding_);
    auto end = std::chrono::high_resolution_clock::now();

    if (!isWarmup) {
      std::chrono::duration<double> duration_seconds = end - start;
      performance_result_.time_costs.emplace_back(duration_seconds.count());
      performance_result_.total_time_cost += duration_seconds.count();
      if (performance_test_config_.run_config.f_verbose) {
        std::cout << "iteration:" << performance_result_.time_costs.size() << ","
                  << "time_cost:" << performance_result_.time_costs.back() << std::endl;
      }
    }
  }

  inline void RunFixDuration() {
    while (performance_result_.total_time_cost < performance_test_config_.run_config.duration_in_seconds) {
      RunOneIteration();
    }
  }

  inline void RunRepeatedTimes() {
    for (size_t ite = 0; ite < performance_test_config_.run_config.repeated_times; ite++) {
      RunOneIteration();
    }
  }

 private:
  PerformanceResult performance_result_;
  PerformanceTestConfig performance_test_config_;

  std::shared_ptr<Lotus::InferenceSession> session_object_;
  std::unique_ptr<IOBinding> io_binding_;
};
}  // namespace PerfTest
}  // namespace Lotus
