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
#include <core/framework/IOBinding.h>

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
  std::cout << "lotus_perf_test -m model_path -r result_file [-t repeated_times] [-d count_of_dataset_to_use] [-e cpu|cuda] [-p profile_file]" << std::endl;
}

struct PerfMetrics {
  size_t peak_workingset_size{0};
  size_t valid_runs{0};
  std::vector<double> time_costs;

  void DumpToFile(const std::string& path, const std::string& model_name) {
    std::ofstream outfile;
    outfile.open(path, std::ofstream::out | std::ofstream::app);
    if (!outfile.good()) {
      LOGF_DEFAULT(ERROR, "failed to open result file");
      return;
    }
    for (size_t runs = 0; runs < valid_runs; runs++) {
      outfile << model_name << "," << time_costs[runs] << "," << peak_workingset_size << "," << runs << std::endl;
    }
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

void RunPerfTest(ITestCase& test_case,
                 PerfMetrics& perf_metrics,
                 size_t repeated_times,
                 size_t count_of_dataset_to_use,
                 Lotus::InferenceSession* session_object,
                 IOBinding* io_binding) {
  size_t data_count = test_case.GetDataCount();

  if (data_count == 0) {
    LOGF_DEFAULT(ERROR, "there is no test data for model %s", test_case.GetTestCaseName().c_str());
    return;
  }

  perf_metrics.time_costs.resize(repeated_times * std::min(data_count, count_of_dataset_to_use));
  size_t count_of_inferences = 0;
  for (size_t times = 0; times < repeated_times; times++) {
    for (size_t data_index = 0; data_index < std::min(data_count, count_of_dataset_to_use); data_index++) {
      std::unordered_map<std::string, Lotus::MLValue> feeds;
      test_case.LoadInputData(data_index, feeds);
      for (auto feed : feeds) {
        io_binding->BindInput(feed.first, feed.second);
      }
      auto outputs = session_object->GetOutputs();
      auto status = outputs.first;
      if (!outputs.first.IsOK()) {
        LOGF_DEFAULT(ERROR, "GetOutputs failed, TestCaseName:%s, ErrorMessage:%s, DataSetIndex:%zu",
                     test_case.GetTestCaseName().c_str(),
                     status.ErrorMessage().c_str(),
                     data_index);
        continue;
      }
      std::vector<MLValue> output_mlvalues(outputs.second->size());
      for (size_t i_output = 0; i_output < outputs.second->size(); ++i_output) {
        auto output = outputs.second->at(i_output);
        if (!output) continue;
        io_binding->BindOutput(output->Name(), output_mlvalues[i_output]);
      }
      TIME_SPEC start_time, end_time;
      GetMonotonicTimeCounter(&start_time);
      status = session_object->Run(*io_binding);
      GetMonotonicTimeCounter(&end_time);
      if (!status.IsOK()) {
        LOGF_DEFAULT(ERROR, "inference failed, TestCaseName:%s, ErrorMessage:%s, DataSetIndex:%zu",
                     test_case.GetTestCaseName().c_str(),
                     status.ErrorMessage().c_str(),
                     data_index);
        continue;
      }
      TIME_SPEC time_cost;
      Lotus::SetTimeSpecToZero(&time_cost);
      AccumulateTimeSpec(&time_cost, &start_time, &end_time);
      perf_metrics.time_costs[count_of_inferences] = TimeSpecToSeconds(&time_cost);
      if(count_of_inferences%100 == 0)
        std::cout << test_case.GetTestCaseName() << ","
                  << perf_metrics.time_costs[count_of_inferences] << ","
                  << count_of_inferences << std::endl;
      count_of_inferences++;
    }
  }

  perf_metrics.valid_runs = count_of_inferences;
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

  size_t repeated_times = 100;
  if (parser.GetCommandArg("-t")) {
    long repeated_times_from_arg = strtol(parser.GetCommandArg("-t")->c_str(), nullptr, 10);
    if (repeated_times_from_arg <= 0) {
      LOGF_DEFAULT(ERROR, "repeated times should be bigger than 0");
      usage();
      return -1;
    }

    repeated_times = static_cast<size_t>(repeated_times_from_arg);
  }

  size_t count_of_dataset_to_use = 1;
  if (parser.GetCommandArg("-d")) {
    long count_of_dataset_to_use_from_arg = strtol(parser.GetCommandArg("-t")->c_str(), nullptr, 10);
    if (count_of_dataset_to_use_from_arg <= 0) {
      LOGF_DEFAULT(ERROR, "count of the dataset to use should be bigger than 0");
      usage();
      return -1;
    }
    count_of_dataset_to_use = static_cast<size_t>(count_of_dataset_to_use_from_arg);
  }

  std::string provider_type = LotusIR::kCpuExecutionProvider;
  if (parser.GetCommandArg("-e")) {
    const std::string& xp_type = *parser.GetCommandArg("-e");
    if (xp_type == "cuda")
      provider_type = LotusIR::kCudaExecutionProvider;
  }

  std::string profile_file;
  if (parser.GetCommandArg("-p")) {
    profile_file = *parser.GetCommandArg("-p");
  }

  std::string model_name = model_path.parent_path().filename().string();
  if (model_name.compare(0, 5, "test_") == 0) model_name = model_name.substr(5);

  OnnxTestCase test_case(model_name);

  if (!test_case.SetModelPath(model_path).IsOK()) {
    LOGF_DEFAULT(ERROR, "load data from %s failed", status.ErrorMessage().c_str());
    return -1;
  }

  std::shared_ptr<Lotus::InferenceSession> session_object;
  SessionFactory sf(provider_type, true, true);
  sf.create(session_object, test_case.GetModelUrl(), test_case.GetTestCaseName());

  if (!profile_file.empty())
    session_object->StartProfiling(profile_file);

  std::unique_ptr<IOBinding> io_binding;
  if (!session_object->NewIOBinding(&io_binding).IsOK()) {
    LOGF_DEFAULT(ERROR, "Failed to init session and IO binding");
    return -1;
  }
  AllocatorPtr cpu_allocator = io_binding->GetCPUAllocator(provider_type);
  test_case.SetAllocator(cpu_allocator);

  PerfMetrics perf_metrics;
  RunPerfTest(test_case, perf_metrics, repeated_times, count_of_dataset_to_use, session_object.get(), io_binding.get());
  perf_metrics.DumpToFile(resultfile, test_case.GetTestCaseName());

  if (!profile_file.empty())
    session_object->EndProfiling();

  return 0;
}
