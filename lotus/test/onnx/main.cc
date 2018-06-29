#include <core/framework/environment.h>
#include <onnx/onnx_pb.h>
#include <core/graph/model.h>
#include <core/framework/allocator.h>
#include <core/framework/op_kernel.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/framework/inference_session.h>
#include <core/common/logging/logging.h>
#include <core/platform/env.h>
#include <core/providers/cpu/cpu_execution_provider.h>
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

using namespace std::experimental::filesystem::v1;
using namespace LotusIR;
using namespace Lotus;

namespace {
void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously.\n"
      "\t-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.\n"
      "\t-n [test_case_name]: Specifies a single test case to run.\n"
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu' or 'cuda'. Default: 'cpu'.\n"
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
  std::string provider = kCpuExecutionProvider;
  //if this var is not empty, only run the tests with name in this list
  std::vector<std::string> whitelisted_test_cases;
  int concurrent_session_runs = Env::Default().GetNumCpuCores();
  int p_models = Env::Default().GetNumCpuCores();
  {
    int ch;
    while ((ch = getopt(argc, argv, "c:hj:m:n:e:")) != -1) {
      switch (ch) {
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
            provider = kCpuExecutionProvider;
          } else if (!strcmp(optarg, "cuda")) {
            provider = kCudaExecutionProvider;
          } else {
            usage();
            return -1;
          }
          break;
        case '?':
        case 'h':
        default:
          usage();
          return -1;
      }
    }
  }
  argc -= optind;
  argv += optind;
  if (argc < 1) {
    fprintf(stderr, "please specify a test data dir\n");
    usage();
    return -1;
  }
  std::vector<string> data_dirs;
  for (int i = 0; i != argc; ++i) {
    path p(argv[i]);
    if (!is_directory(p)) {
      fprintf(stderr, "input dir %s is not a valid directoy", argv[i]);
      return -1;
    }
    data_dirs.emplace_back(argv[i]);
  }
  AllocatorPtr cpu_allocator(new Lotus::CPUAllocator());
  std::vector<ITestCase*> tests = LoadTests(data_dirs, whitelisted_test_cases, cpu_allocator);
  TestResultStat stat;
  SessionFactory sf(provider);
  TestEnv args(tests, stat, sf);
  RunTests(args, p_models, concurrent_session_runs);
  std::string res = stat.ToString();
  fwrite(res.c_str(), 1, res.size(), stdout);
  for (ITestCase* l : tests) {
    delete l;
  }
  std::unordered_set<std::string> broken_tests{"cast_DOUBLE_to_FLOAT", "cast_DOUBLE_to_FLOAT16",
                                               "AvgPool1d", "AvgPool1d_stride", "AvgPool2d", "AvgPool2d_stride", "AvgPool3d", "AvgPool3d_stride",
                                               "AvgPool3d_stride1_pad0_gpu_input", "BatchNorm1d_3d_input_eval", "BatchNorm2d_eval", "BatchNorm2d_momentum_eval",
                                               "BatchNorm3d_eval", "BatchNorm3d_momentum_eval",
                                               "GLU", "GLU_dim", "Linear", "not_2d", "not_3d", "not_4d", "operator_add_broadcast",
                                               "operator_add_size1_broadcast", "operator_add_size1_right_broadcast", "operator_add_size1_singleton_broadcast",
                                               "operator_addconstant", "operator_addmm", "operator_basic", "operator_lstm", "operator_mm", "operator_non_float_params",
                                               "operator_params", "operator_pow", "operator_rnn", "operator_rnn_single_layer", "PoissonNLLLLoss_no_reduce", "PReLU_1d",
                                               "PReLU_1d_multiparam", "PReLU_2d", "PReLU_2d_multiparam", "PReLU_3d", "PReLU_3d_multiparam", "Softsign"};
  for (const std::string s : stat.GetFailedTest()) {
    if (broken_tests.find(s) == broken_tests.end()) return -1;
  }
  return 0;
}
