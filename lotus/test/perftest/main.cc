#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif

#include <fstream>
#include <iostream>

// Lotus dependencies
#include <core/common/logging/sinks/clog_sink.h>
#include <core/common/logging/logging.h>
#include <core/framework/environment.h>
#include <core/framework/inference_session.h>
#include <core/platform/env.h>
#include <core/providers/cpu/cpu_execution_provider.h>

// Windows Specific
#ifdef _WIN32
#include "windows.h"
#include "psapi.h"
#endif

#include "CmdParser.h"
#include "runner.h"

using namespace std::experimental::filesystem::v1;
using namespace LotusIR;
using namespace Lotus;

void usage() {
  std::cout << "lotus_perf_test -m model_path -r result_file" << std::endl;
}

struct PerfMetrics {
  double seconds_spent;
  size_t peak_workingset_size;

  void DumpToFile(const std::string& path, const std::string& model_name) {
    std::ofstream outfile;
    outfile.open(path, std::ofstream::out | std::ofstream::app);
    if( !outfile.good() ){
      LOGF_DEFAULT(ERROR, "failed to open result file");
      return;
    }
    outfile << model_name << "," << seconds_spent << "," << peak_workingset_size << std::endl;
    outfile.close();
  }
};

size_t GetPeakWorkingSetSize() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
     return pmc.PeakWorkingSetSize;
  } else {
    LOGF_DEFAULT(ERROR, "failed to get process memory");
    return 0;
  }
#else
  return 0;
#endif
}

void RunPerfTest(ITestCase& test_case, PerfMetrics& perf_metrics) {
  std::shared_ptr<Lotus::InferenceSession> session_object;
  SessionFactory sf(kCpuExecutionProvider);
  sf.create(session_object, test_case.GetModelUrl(), test_case.GetTestCaseName());

  size_t data_count = test_case.GetDataCount();
  TIME_SPEC time_spent;
  Lotus::SetTimeSpecToZero(&time_spent);

  for (size_t data_index = 0; data_index < data_count; data_index++){
    std::unordered_map<std::string, Lotus::MLValue> feeds;
    test_case.LoadInputData(data_index, feeds);
    std::vector<MLValue> fetches;
    TIME_SPEC start_time, end_time;
    GetMonotonicTimeCounter(&start_time);
    Status status = session_object->Run(feeds, &fetches);
    if (!status.IsOK()) {
      LOGF_DEFAULT(ERROR, "inference failed, TestCaseName:%s, ErrorMessage:%s, DataSetIndex:%zu", test_case.GetTestCaseName().c_str(), status.ErrorMessage().c_str(), data_index);
       continue;
    }
    GetMonotonicTimeCounter(&end_time);
    AccumulateTimeSpec(&time_spent, &start_time, &end_time);
  }

  perf_metrics.seconds_spent = TimeSpecToSeconds(&time_spent) / data_count;
  perf_metrics.peak_workingset_size = GetPeakWorkingSetSize();
}

int main(int argc, const char* args[]) {
  std::string default_logger_id{"Default"};
  Logging::LoggingManager default_logging_manager{std::unique_ptr<Logging::ISink>{new Logging::CLogSink{}},
                                                  Logging::Severity::kWARNING, false,
                                                  Logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  std::unique_ptr<Environment> env;
  auto status = Environment::Create(env);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "failed to create environment:%s", status.ErrorMessage().c_str());
    return -1;
  }

  CmdParser parser(argc, args);
  if (!parser.GetCommandArg("-m")) {
    LOGF_DEFAULT(ERROR, "model path is empty.");
    usage();
    return -1;
  }
  path model_path(*parser.GetCommandArg("-m"));
  if (model_path.extension() != ".onnx") {
    LOGF_DEFAULT(ERROR, "input path is not a valid model");
    return -1;
  }

  if (!parser.GetCommandArg("-r")) {
    LOGF_DEFAULT(ERROR, "result file is empty");
    usage();
    return -1;
  }
  const std::string resultfile = *parser.GetCommandArg("-r");

  std::string model_name = model_path.parent_path().filename().string();
  if (model_name.compare(0, 5, "test_") == 0) model_name = model_name.substr(5);

  AllocatorPtr cpu_allocator(new Lotus::CPUAllocator());
  OnnxTestCase test_case(cpu_allocator, model_name);
  if (!test_case.SetModelPath(model_path).IsOK()) {
    LOGF_DEFAULT(ERROR, "load data from %s failed", status.ErrorMessage().c_str());
    return -1;
  }

  PerfMetrics perf_metrics;
  RunPerfTest(test_case, perf_metrics);
  perf_metrics.DumpToFile(resultfile, test_case.GetTestCaseName());
  std::cout << test_case.GetTestCaseName() << "," << perf_metrics.seconds_spent << "," << perf_metrics.peak_workingset_size << std::endl;

  return 0;
}
