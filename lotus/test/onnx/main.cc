#include <core/framework/environment.h>
//#include <onnx/onnx-ml.pb.h>
#include "onnx/onnx_pb.h"
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

#include "TestCaseInfo.h"
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
      "\t-p [PLANNER_TYPE]: PLANNER_TYPE could be 'seq' or 'simple'. Default: 'simple'.\n"
      "\t-h: help\n");
}

}  // namespace

// Create an object to shutdown the protobuf library at exit
struct ShutdownProtobufs {
  ~ShutdownProtobufs() {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
} s_shutdown_protobufs;

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
  AllocationPlannerType planner = AllocationPlannerType::SEQUENTIAL_PLANNER;
  //if this var is not empty, only run the tests with name in this list
  std::vector<std::string> whitelisted_test_cases;
  int concurrent_session_runs = Env::Default()->GetNumCpuCores();
  int p_models = Env::Default()->GetNumCpuCores();
  {
    int ch;
    while ((ch = getopt(argc, argv, "c:hj:m:n:p:")) != -1) {
      switch (ch) {
        case 'c':
          concurrent_session_runs = (int)strtol(optarg, NULL, 10);
          if (concurrent_session_runs <= 0) {
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
          whitelisted_test_cases.push_back(optarg);
          break;
        case 'p':
          if (!strcmp(optarg, "simple")) {
            planner = AllocationPlannerType::SIMPLE_SEQUENTIAL_PLANNER;
          } else if (!strcmp(optarg, "seq")) {
            planner = AllocationPlannerType::SEQUENTIAL_PLANNER;
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
    data_dirs.push_back(argv[i]);
  }
  std::vector<TestCaseInfo> tests = LoadTests(data_dirs, whitelisted_test_cases);
  TestResultStat stat;
  TestEnv args(tests, stat, planner);
  RunTests(args, p_models, concurrent_session_runs);
  std::string res = stat.ToString();
  fwrite(res.c_str(), 1, res.size(), stdout);

  return 0;
}
