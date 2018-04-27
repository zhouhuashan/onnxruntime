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
  exit(-1);
}

void RunTests(TestEnv& env, int p_models, int concurrent_runs) {
  TestResultStat& stat = env.stat;
  stat.total_test_case_count = std::accumulate(env.tests.begin(), env.tests.end(), (size_t)0, [](size_t v, const TestCaseInfo& info) {
    return info.input_pb_files.size() + v;
  });
  std::vector<TestCaseResult> results(env.tests.size());
#ifdef _WIN32
  if (p_models > 1) {
    ParallelRunTests(env, p_models, concurrent_runs, results);
  } else
#endif
  {
    for (size_t i = 0; i != env.tests.size(); ++i) {
      RunSingleTestCase(env, i, concurrent_runs, [i, &results](TestCaseResult& result) {
        results[i] = result;
      });
    }
  }
  for (size_t i = 0; i != env.tests.size(); ++i) {
    const TestCaseResult& r = results[i];
    for (const EXECUTE_RESULT res : r.excution_result) {
      if (res != EXECUTE_RESULT::SUCCESS && res != EXECUTE_RESULT::NOT_SUPPORT) {
        stat.AddFailedTest(env.tests[i].test_case_name);
      }
      switch (res) {
        case EXECUTE_RESULT::SUCCESS:
          stat.succeeded++;
          break;
        case EXECUTE_RESULT::UNKNOWN_ERROR:
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::INVALID_GRAPH:
          stat.invalid_graph++;
          break;
        case EXECUTE_RESULT::WITH_EXCEPTION:
          stat.throwed_exception++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::RESULT_DIFFERS:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::SHAPE_MISMATCH:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::TYPE_MISMATCH:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::NOT_SUPPORT:
          stat.not_implemented++;
          if (!r.node_name.empty()) stat.AddNotImplementedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::LOAD_MODEL_FAILED:
          stat.load_model_failed++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        default:
          abort();
      }
    }
  }
}

int ExtractFileNo(const std::string& name) {
  size_t p1 = name.rfind('.');
  size_t p2 = name.rfind('_', p1);
  ++p2;
  std::string number_str = name.substr(p2, p1 - p2);
  const char* start = number_str.c_str();
  const char* end = number_str.c_str();
  long ret = strtol(start, (char**)&end, 10);
  if (end == start) {
    LOTUS_THROW("parse file name failed");
  }
  return (int)ret;
}

void SortTensorFileNames(std::vector<path>& input_pb_files) {
  std::sort(input_pb_files.begin(), input_pb_files.end(), [](const path& left, const path& right) -> bool {
    std::string leftname = left.filename().string();
    std::string rightname = right.filename().string();
    int left1 = ExtractFileNo(leftname);
    int right1 = ExtractFileNo(rightname);
    return left1 < right1;
  });
  for (size_t i = 0; i != input_pb_files.size(); ++i) {
    int fileno = ExtractFileNo(input_pb_files[i].filename().string());
    if (fileno != i) {
      LOTUS_THROW("illegal input file names");
    }
  }
}

/**
* test_case_dir must have contents of:
* model.onnx
* ???/input_??.pb
* ???/output_??.pb
* ???/input_??.pb
* ???/output_??.pb
*/
TestCaseInfo GatherTests(const std::string& test_case_name, const path& test_case_dir) {
  const std::string model_file_path = (test_case_dir / "model.onnx").string();
  const path pb(".pb");
  TestCaseInfo info;
  info.test_case_name = test_case_name;
  info.model_url = model_file_path;
  for (directory_iterator test_data_set(test_case_dir), end2; test_data_set != end2; ++test_data_set) {
    if (!is_directory(*test_data_set)) {
      continue;
    }
    std::vector<path> inputs;
    std::vector<path> outputs;
    for (directory_iterator pb_file(*test_data_set), end3; pb_file != end3; ++pb_file) {
      path f = *pb_file;
      if (!is_regular_file(f)) continue;
      if (f.extension() != pb) continue;
      std::string filename = f.filename().string();
      if (!filename.compare(0, 6, "input_")) {
        inputs.push_back(f);
      } else if (!filename.compare(0, 7, "output_")) {
        outputs.push_back(f);
      }
    }
    SortTensorFileNames(inputs);
    SortTensorFileNames(outputs);
    info.input_pb_files.push_back(inputs);
    info.output_pb_files.push_back(outputs);
  }
  return info;
}
std::vector<TestCaseInfo> LoadTests(const std::vector<path>& input_paths, const std::vector<std::string>& whitelisted_test_cases) {
  std::vector<TestCaseInfo> tests;
  for (const path& test_data_root_path : input_paths) {
    path node_data_root_path = test_data_root_path;
    for (directory_iterator test_case_dir(node_data_root_path), end; test_case_dir != end; ++test_case_dir) {
      if (!is_directory(*test_case_dir)) {
        continue;
      }
      std::string test_dir_name = test_case_dir->path().filename().string();
      if (test_dir_name.compare(0, 5, "test_")) continue;
      std::string test_case_name = test_dir_name.substr(5);
      if (!whitelisted_test_cases.empty() && std::find(whitelisted_test_cases.begin(), whitelisted_test_cases.end(), test_case_name) == whitelisted_test_cases.end()) {
        continue;
      }
      tests.emplace_back(GatherTests(test_case_name, test_case_dir->path()));
    }
  }
  return tests;
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
  std::vector<path> data_dirs;
  for (int i = 0; i != argc; ++i) {
    path p(argv[i]);
    if (!is_directory(p)) {
      fprintf(stderr, "input dir %s is not a valid directoy", argv[i]);
      return -1;
    }
    data_dirs.push_back(p);
  }
  std::vector<TestCaseInfo> tests = LoadTests(data_dirs, whitelisted_test_cases);
  TestResultStat stat;
  std::vector<std::string> all_implemented_ops = KernelRegistry::Instance().GetAllRegisteredOpNames();
  TestEnv args(tests, all_implemented_ops, stat, planner);
  RunTests(args, p_models, concurrent_session_runs);
  stat.print(stdout);
  return 0;
}
