// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/framework/environment.h>
#include <core/graph/constants.h>
#include <core/framework/allocator.h>
#include <core/common/logging/logging.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/session/inference_session.h>
#include <core/platform/env.h>
#include <iostream>
#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif
#include <fstream>
#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#endif

#include "TestResultStat.h"
#include "testenv.h"
#include "runner.h"
#include "sync_api.h"

using namespace std::experimental::filesystem::v1;
using namespace onnxruntime;

namespace {
void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously.\n"
      "\t-A : Disable memory arena\n"
      "\t-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.\n"
      "\t-r [repeat]: Specifies the number of times to repeat\n"
      "\t-n [test_case_name]: Specifies a single test case to run.\n"
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu' or 'cuda'. Default: 'cpu'.\n"
      "\t-x: Use parallel executor, default (without -x): sequential executor.\n"
      "\t-h: help\n");
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string default_logger_id{"Default"};
  Logging::LoggingManager default_logging_manager{std::unique_ptr<Logging::ISink>{new Logging::CLogSink{}},
                                                  Logging::Severity::kWARNING, false,
                                                  Logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  std::unique_ptr<Environment> env;
  auto status = Environment::Create(env);
  if (!status.IsOK()) {
    fprintf(stderr, "Error creating environment: %s \n", status.ErrorMessage().c_str());
    return -1;
  }

  std::vector<std::string> providers{onnxruntime::kCpuExecutionProvider};
  //if this var is not empty, only run the tests with name in this list
  std::vector<std::string> whitelisted_test_cases;
  int concurrent_session_runs = Env::Default().GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  bool enable_sequential_execution = true;
  int repeat_count = 1;
  int p_models = Env::Default().GetNumCpuCores();
  {
    int ch;
    while ((ch = getopt(argc, argv, "Ac:hj:m:n:r:e:x")) != -1) {
      switch (ch) {
        case 'A':
          enable_cpu_mem_arena = false;
          break;
        case 'c':
          concurrent_session_runs = static_cast<int>(strtol(optarg, nullptr, 10));
          if (concurrent_session_runs <= 0) {
            usage();
            return -1;
          }
          break;
        case 'j':
          p_models = static_cast<int>(strtol(optarg, nullptr, 10));
          if (p_models <= 0) {
            usage();
            return -1;
          }
          break;
        case 'r':
          repeat_count = static_cast<int>(strtol(optarg, nullptr, 10));
          if (repeat_count <= 0) {
            usage();
            return -1;
          }
          break;
        case 'm':
          //ignore.
          break;
        case 'n':
          //run only some whitelisted tests
          //TODO: parse name str to an array
          whitelisted_test_cases.emplace_back(optarg);
          break;
        case 'e':
          if (!strcmp(optarg, "cpu")) {
            providers.push_back(onnxruntime::kCpuExecutionProvider);
          } else if (!strcmp(optarg, "cuda")) {
            providers.push_back(onnxruntime::kCudaExecutionProvider);
          } else {
            usage();
            return -1;
          }
          break;
        case 'x':
          enable_sequential_execution = false;
          break;
        case '?':
        case 'h':
        default:
          usage();
          return -1;
      }
    }
  }
  if (concurrent_session_runs > 1 && repeat_count > 1) {
    fprintf(stderr, "when you use '-r [repeat]', please set '-c' to 1\n");
    usage();
    return -1;
  }
  argc -= optind;
  argv += optind;
  if (argc < 1) {
    fprintf(stderr, "please specify a test data dir\n");
    usage();
    return -1;
  }
  std::vector<path> data_dirs;
  for (int i = 0; i != argc; ++i) {
    path p(argv[i]);
    if (!is_directory(p)) {
      fprintf(stderr, "input dir %s is not a valid directoy\n", argv[i]);
      return -1;
    }
    data_dirs.emplace_back(p);
  }
  AllocatorPtr cpu_allocator = std::make_shared<::onnxruntime::CPUAllocator>();
  std::vector<ITestCase*> tests = LoadTests(data_dirs, whitelisted_test_cases, cpu_allocator);
  TestResultStat stat;
  SessionFactory sf(providers, true, enable_cpu_mem_arena);
  sf.enable_sequential_execution = enable_sequential_execution;
  TestEnv args(tests, stat, sf);
  Status st = RunTests(args, p_models, concurrent_session_runs, static_cast<size_t>(repeat_count), GetDefaultThreadPool(Env::Default()));
  if (!st.IsOK()) {
    fprintf(stderr, "%s\n", st.ErrorMessage().c_str());
    return -1;
  }

  std::string res = stat.ToString();
  fwrite(res.c_str(), 1, res.size(), stdout);
  for (ITestCase* l : tests) {
    delete l;
  }

  std::unordered_set<std::string> broken_tests{"cast_DOUBLE_to_FLOAT", "cast_DOUBLE_to_FLOAT16",
                                               "AvgPool1d", "AvgPool1d_stride", "AvgPool2d", "AvgPool2d_stride", "AvgPool3d", "AvgPool3d_stride",
                                               "AvgPool3d_stride1_pad0_gpu_input", "BatchNorm1d_3d_input_eval", "BatchNorm2d_eval", "BatchNorm2d_momentum_eval",
                                               "BatchNorm3d_eval", "BatchNorm3d_momentum_eval",
                                               "GLU", "GLU_dim", "Linear", "operator_add_broadcast",
                                               "operator_add_size1_broadcast", "operator_add_size1_right_broadcast", "operator_add_size1_singleton_broadcast",
                                               "operator_addconstant", "operator_addmm", "operator_basic", "operator_lstm", "operator_mm", "operator_non_float_params",
                                               "operator_params", "operator_pow", "operator_rnn", "operator_rnn_single_layer", "PoissonNLLLLoss_no_reduce", "PReLU_1d",
                                               "PReLU_1d_multiparam", "PReLU_2d", "PReLU_2d_multiparam", "PReLU_3d", "PReLU_3d_multiparam", "Softsign", "min_one_input",
                                               "sum_two_inputs", "mvn", "max_example", "maxpool_2d_default", "sum_one_input", "sum_example", "min_two_inputs", "min_example", "mean_two_inputs",
                                               "maxpool_1d_default", "mean_one_input", "mean_example", "max_two_inputs", "max_one_input", "maxpool_2d_pads", "maxpool_with_argmax_2d_precomputed_strides",
                                               "maxpool_with_argmax_2d_precomputed_pads", "maxpool_3d_default", "maxpool_2d_strides", "maxpool_2d_same_upper", "maxpool_2d_same_lower",
                                               "maxpool_2d_precomputed_strides", "maxpool_2d_precomputed_same_upper", "maxpool_2d_precomputed_pads", "expand_dim_unchanged",
                                               "convtranspose_1d", "expand_dim_changed", "convtranspose_3d", "expand_shape_model1", "expand_shape_model2", "expand_shape_model3", "expand_shape_model4"};

  int result = 0;
  for (const std::string& s : stat.GetFailedTest()) {
    if (broken_tests.find(s) == broken_tests.end()) {
      fprintf(stderr, "test %s failed, please fix it\n", s.c_str());
      result = -1;
    }
  }

  return result;
}
