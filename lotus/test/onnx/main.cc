#include <core/framework/environment.h>
//#include <onnx/onnx-ml.pb.h>
#include "onnx/onnx_pb.h"
#include <core/graph/model.h>
#include <core/framework/allocator.h>
#include <core/framework/op_kernel.h>
#include <core/framework/tensorprotoutils.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/framework/inference_session.h>
#include <core/common/logging/logging.h>
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
#include "sysutil.h"
#include "runner.h"

using namespace std::experimental::filesystem::v1;
using namespace LotusIR;
using namespace Lotus;

namespace {
inline bool IsTheSameType(MLDataType left, onnx::TensorProto_DataType right) {
  return DataTypeImpl::ElementTypeFromProto(right) == left;
}

MLValue ConvertTensorProtoToMLValue(const onnx::TensorProto& input, IAllocator& allocator) {
  std::unique_ptr<Tensor> tensor;
  LOTUS_ENFORCE(Lotus::Utils::GetTensorFromTensorProto(input, &tensor, allocator).IsOK());
  MLValue value;
  value.Init(tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return value;
}

//If we cannot get input name from input_pbs, we'll use names like "data_0","data_1",... It's dirty hack
// for https://github.com/onnx/onnx/issues/679
std::unordered_map<std::string, MLValue> ConvertPbsToMLValues(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& input_value_info, const std::vector<onnx::TensorProto>& input_pbs, IAllocator& allocator) {
  std::unordered_map<std::string, MLValue> feeds;
  int len = input_value_info.size();
  bool has_valid_names = true;
  //"0","1",...
  bool use_number_names = true;
  //"data_0","data_1",...
  bool use_data_number_names = true;
  //"gpu_0/data_0","gpu_0/data_1",...
  bool use_gpu_data_number_names = true;

  std::vector<std::string> var_names(input_pbs.size());
  for (size_t input_index = 0; input_index != input_pbs.size(); ++input_index) {
    std::string name = input_pbs[input_index].name();
    if (name.empty()) {
      has_valid_names = false;
      break;
    }
    var_names[input_index] = name;
  }
  if (!has_valid_names) {
    if (len == input_pbs.size()) {
      for (int i = 0; i != len; ++i) {
        std::string vname = input_value_info[i].name();
        var_names[i] = vname;
      }
    } else {
      char buf[64];
      char buf2[64];
      char buf3[64];
      for (int i = 0; i != input_pbs.size(); ++i) {
        snprintf(buf, sizeof(buf), "%d", i);
        snprintf(buf2, sizeof(buf2), "data_%d", i);
        snprintf(buf3, sizeof(buf3), "gpu_0/data_%d", i);
        if (use_number_names && std::find_if(input_value_info.begin(), input_value_info.end(), [buf](const onnx::ValueInfoProto& info) {
                                  return info.name() == buf;
                                }) == input_value_info.end()) use_number_names = false;
        if (use_data_number_names && std::find_if(input_value_info.begin(), input_value_info.end(), [buf2](const onnx::ValueInfoProto& info) {
                                       return info.name() == buf2;
                                     }) == input_value_info.end()) use_data_number_names = false;
        if (use_data_number_names && std::find_if(input_value_info.begin(), input_value_info.end(), [buf3](const onnx::ValueInfoProto& info) {
                                       return info.name() == buf3;
                                     }) == input_value_info.end()) use_gpu_data_number_names = false;
      }
    }
    for (size_t input_index = 0; input_index != input_pbs.size(); ++input_index) {
      std::string name = var_names[input_index];
      char buf[64];
      if (name.empty()) {
        if (use_number_names) {
          snprintf(buf, sizeof(buf), "%d", (int)input_index);
          var_names[input_index] = buf;
        } else if (use_data_number_names) {
          snprintf(buf, sizeof(buf), "data_%d", (int)input_index);
          var_names[input_index] = buf;
        } else if (use_gpu_data_number_names) {
          snprintf(buf, sizeof(buf), "gpu_0/data_%d", (int)input_index);
          var_names[input_index] = buf;
        } else
          LOTUS_THROW("cannot guess a valid input name");
      }
    }
  }
  for (size_t input_index = 0; input_index != input_pbs.size(); ++input_index) {
    std::string name = var_names[input_index];
    const onnx::TensorProto& input = input_pbs[input_index];
    feeds.insert(std::make_pair(name, ConvertTensorProtoToMLValue(input, allocator)));
  }
  return feeds;
}

template <typename FLOAT_TYPE>
std::pair<EXECUTE_RESULT, size_t> compare_float_result(const Tensor& outvalue, const Tensor& expected_value) {
  const FLOAT_TYPE abs_error = 1e-6f;
  if (expected_value.Shape() != outvalue.Shape()) return std::make_pair(EXECUTE_RESULT::SHAPE_MISMATCH, -1);
  const size_t size1 = expected_value.Shape().Size();
  const FLOAT_TYPE* expected_output = expected_value.Data<FLOAT_TYPE>();
  const FLOAT_TYPE* real_output = outvalue.Data<FLOAT_TYPE>();
  for (size_t di = 0; di != size1; ++di) {
    const double diff = fabs(expected_output[di] - real_output[di]);
    if (diff > abs_error) {
      return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, di);
    }
  }
  return std::make_pair(EXECUTE_RESULT::SUCCESS, -1);
}

template <typename T>
std::pair<EXECUTE_RESULT, size_t> is_result_exactly_match(const Tensor& outvalue, const Tensor& expected_value) {
  if (expected_value.Shape() != outvalue.Shape()) return std::make_pair(EXECUTE_RESULT::SHAPE_MISMATCH, -1);
  const size_t size1 = expected_value.Shape().Size();
  const T* expected_output = expected_value.Data<T>();
  const T* real_output = outvalue.Data<T>();
  for (size_t di = 0; di != size1; ++di) {
    if (expected_output[di] != real_output[di]) {
      return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, di);
    }
  }
  return std::make_pair(EXECUTE_RESULT::SUCCESS, -1);
}

std::pair<EXECUTE_RESULT, size_t> compare(const Tensor& outvalue, const onnx::TensorProto& expected_value, IArenaAllocator& cpu_allocator) {
  std::unique_ptr<Tensor> expected_tensor;
  LOTUS_ENFORCE(Lotus::Utils::GetTensorFromTensorProto(expected_value, &expected_tensor, cpu_allocator).IsOK());
  switch (expected_value.data_type()) {
    case TensorProto_DataType_FLOAT:
      return compare_float_result<float>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_UINT8:
      return is_result_exactly_match<uint8_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_INT8:
      return is_result_exactly_match<int8_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_UINT16:
      return is_result_exactly_match<uint16_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_INT16:
      return is_result_exactly_match<int16_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_INT32:
      return is_result_exactly_match<int32_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_INT64:
      return is_result_exactly_match<int64_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_BOOL:
      return is_result_exactly_match<bool>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_DOUBLE:
      return compare_float_result<double>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_UINT32:
      return is_result_exactly_match<uint32_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_UINT64:
      return is_result_exactly_match<uint64_t>(outvalue, *expected_tensor.get());
      break;
    case TensorProto_DataType_STRING:
    case TensorProto_DataType_FLOAT16:
    case TensorProto_DataType_COMPLEX64:
    case TensorProto_DataType_COMPLEX128:
    default:
      LOTUS_NOT_IMPLEMENTED;
  }
}

EXECUTE_RESULT ExecuteModelWithProtobufs(InferenceSession& sess, const std::vector<onnx::TensorProto>& input_pbs,
                                         const std::vector<onnx::TensorProto>& output_pbs, const char* test_case_name,
                                         const onnx::ModelProto& model_pb) {
  auto& cpu_allocator = AllocatorManager::Instance().GetArena(CPU);
  std::unordered_map<std::string, MLValue> feeds = ConvertPbsToMLValues(model_pb.graph().input(), input_pbs, cpu_allocator);
  std::vector<MLValue> p_fetches;
  try {
    Common::Status status = sess.Run(feeds, &p_fetches);
    if (!status.IsOK()) {
      fprintf(stderr, "%s:%s\n", test_case_name, status.ErrorMessage().c_str());
      return EXECUTE_RESULT::FAILED_TO_RUN;
    }
  } catch (std::exception& ex) {
    fprintf(stderr, "%s:%s\n", test_case_name, ex.what());
    return EXECUTE_RESULT::FAILED_TO_RUN;
  } catch (...) {
    fprintf(stderr, "%s:got unknown error\n", test_case_name);
    return EXECUTE_RESULT::FAILED_TO_RUN;
  }

  for (size_t i = 0; i != output_pbs.size(); ++i) {
    const Tensor& outvalue = p_fetches.at(i).Get<Tensor>();
    const onnx::TensorProto& expected_value = output_pbs.at(i);
    if (!IsTheSameType(outvalue.DataType(), expected_value.data_type())) {
      fprintf(stderr, "%s:type mismatch\n", test_case_name);
      return EXECUTE_RESULT::TYPE_MISMATCH;
    }
    //TODO: support comparisons other than float/bool/int32/...
    auto ret = compare(outvalue, expected_value, cpu_allocator);
    EXECUTE_RESULT compare_result = ret.first;
    if (compare_result == EXECUTE_RESULT::RESULT_DIFFERS)
      fprintf(stderr, "%s: the %zd-th value of the %zd-th output differs\n", test_case_name, ret.second, i);
    if (compare_result != EXECUTE_RESULT::SUCCESS)
      return compare_result;
  }
  printf("test %s succeeded\n", test_case_name);
  return EXECUTE_RESULT::SUCCESS;
}

//load tensors from disk
std::vector<onnx::TensorProto> LoadTensors(const std::vector<path>& pb_files) {
  std::vector<onnx::TensorProto> input_pbs;
  for (size_t i = 0; i != pb_files.size(); ++i) {
    onnx::TensorProto tensor;
    std::ifstream input(pb_files.at(i), std::ios::in | std::ios::binary);
    if (!input) {
      LOTUS_THROW("open file failed");
    }
    if (!tensor.ParseFromIstream(&input)) {
      LOTUS_THROW("parse file failed");
    }
    input_pbs.emplace_back(tensor);
  }
  return input_pbs;
}

void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously"
      "\t-m [TEST_MODE]: TEST_MODE could be 'node' or 'model'. Default: 'node'.\n"
      "\t-p [PLANNER_TYPE]: PLANNER_TYPE could be 'seq' or 'simple'. Default: 'simple'.\n"
      "\t-h: help\n");
  exit(-1);
}

bool IsImplemented(const onnx::ModelProto& model_pb, const std::vector<std::string>& all_implemented_ops, TestResultStat& stat) {
  bool ret = true;
  for (const auto& node_pb : model_pb.graph().node()) {
    stat.AddCoveredOps(node_pb.op_type());
    if (std::find(all_implemented_ops.begin(), all_implemented_ops.end(), node_pb.op_type()) == all_implemented_ops.end()) {
      stat.AddNotImplementedKernels(node_pb.op_type());
      ret = false;
    }
  }
  return ret;
}

void RunTests(TestEnv& env, int p_models) {
  TestResultStat& stat = env.stat;
  stat.total_test_case_count = std::accumulate(env.tests.begin(), env.tests.end(), (size_t)0, [](size_t v, const TestCaseInfo& info) {
    return info.input_pb_files.size() + v;
  });
  std::vector<TestCaseResult> results(env.tests.size());
#ifdef _WIN32
  if (p_models > 1) {
    ParallelRunTests(env, p_models, results);
  } else
#endif
  {
    for (size_t i = 0; i != env.tests.size(); ++i) {
      RunSingleTestCase(env, i, [i, &results](TestCaseResult& result) {
        results[i] = result;
      });
    }
  }
  for (const TestCaseResult& r : results) {
    for (const EXECUTE_RESULT res : r.excution_result) {
      switch (res) {
        case EXECUTE_RESULT::SUCCESS:
          stat.succeeded++;
          break;
        case EXECUTE_RESULT::UNKNOWN_ERROR:
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::FAILED_TO_RUN:
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
          stat.skipped++;
          break;
        case EXECUTE_RESULT::LOAD_MODEL_FAILED:
          stat.load_model_failed++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::KERNEL_NOT_IMPLEMENTED:
          stat.not_implemented++;
          if (!r.node_name.empty()) stat.AddNotImplementedKernels(r.node_name);
          break;
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

void RunSingleTestCase(TestEnv& env, size_t test_index, std::function<void(TestCaseResult& result)> on_finished) {
  const TestCaseInfo& info = env.tests[test_index];
  const std::vector<std::string>& all_implemented_ops = env.all_implemented_ops;
  const AllocationPlannerType planner = env.planner;
  onnx::ModelProto model_pb;
  {
    std::ifstream input(info.model_url, std::ios::in | std::ios::binary);
    if (!input) {
      fprintf(stderr, "open file failed");
      TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), ""};
      on_finished(ret);
      return;
    }
    if (!model_pb.ParseFromIstream(&input)) {
      fprintf(stderr, "parse file failed");
      TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), ""};
      on_finished(ret);
      return;
    }
  }
  std::string node_name;
  if (model_pb.graph().node().size() == 1) {
    node_name = model_pb.graph().node()[0].op_type();
  }
  if (!IsImplemented(model_pb, all_implemented_ops, env.stat)) {
    TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::KERNEL_NOT_IMPLEMENTED), node_name};
    on_finished(ret);
    return;
  }
  ExecutionProviderInfo epi;
  ProviderOption po{"CPUExecutionProvider", epi};
  SessionOptions so(vector<ProviderOption>{po});
  so.allocation_planner_type = planner;
  InferenceSession session_object{so};
  Common::Status status = session_object.Load(info.model_url);
  if (!status.IsOK()) {
    fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
    TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), node_name};
    on_finished(ret);
    return;
  }
  try {
    status = session_object.Initialize();
    if (!status.IsOK()) {
      fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
      TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), node_name};
      on_finished(ret);
      return;
    }
  } catch (std::exception& ex) {
    fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), ex.what());
    TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), node_name};
    on_finished(ret);
    return;
  }
  fprintf(stderr, "testing %s\n", info.test_case_name.c_str());
  size_t datasets = info.input_pb_files.size();
  TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::UNKNOWN_ERROR), node_name};
  for (size_t i = 0; i != datasets; ++i) {
    std::vector<onnx::TensorProto> input_pbs = LoadTensors(info.input_pb_files[i]);
    std::vector<onnx::TensorProto> output_pbs = LoadTensors(info.output_pb_files[i]);
    ret.excution_result[i] = ExecuteModelWithProtobufs(session_object, input_pbs, output_pbs, info.test_case_name.c_str(), model_pb);
  }
  on_finished(ret);
}

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
  AllocationPlannerType planner = AllocationPlannerType::SIMPLE_SEQUENTIAL_PLANNER;
  //if this var is not empty, only run the tests with name in this list
  std::vector<std::string> whitelisted_test_cases;
  int p_models = getCoreNum();
  {
    int ch;
    while ((ch = getopt(argc, argv, "hj:m:n:p:")) != -1) {
      switch (ch) {
        case 'j':
          p_models = (int)strtol(optarg, NULL, 10);
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
  RunTests(args, p_models);
  stat.print(all_implemented_ops, !whitelisted_test_cases.empty(), stdout);
  return 0;
}
