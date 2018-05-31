#include "runner.h"
#include <core/common/logging/logging.h>
#include <core/framework/tensorprotoutils.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#ifdef _MSC_VER
#include <filesystem>
#endif
#include <fstream>
#include <cmath>
#include <core/common/logging/logging.h>

#ifdef USE_CUDA
#include <core/providers/cuda/cuda_execution_provider.h>
#endif

#include "FixedCountFinishCallback.h"

using std::experimental::filesystem::v1::directory_iterator;
using std::experimental::filesystem::v1::is_directory;
using std::experimental::filesystem::v1::path;
using namespace Lotus;

namespace {
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

std::pair<EXECUTE_RESULT, size_t> compare(const Tensor& outvalue, const onnx::TensorProto& expected_value, AllocatorPtr cpu_allocator) {
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
      LOTUS_NOT_IMPLEMENTED("Onnx type: ", expected_value.data_type(), " is not supported.");
  }
}

inline bool IsTheSameType(MLDataType left, onnx::TensorProto_DataType right) {
  return DataTypeImpl::ElementTypeFromProto(right) == left;
}

MLValue ConvertTensorProtoToMLValue(const onnx::TensorProto& input, AllocatorPtr allocator) {
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
std::unordered_map<std::string, MLValue> ConvertPbsToMLValues(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& input_value_info, const std::vector<onnx::TensorProto>& input_pbs, AllocatorPtr allocator) {
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
}  // namespace
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

EXECUTE_RESULT StatusCodeToExecuteResult(int input) {
  switch (input) {
    case StatusCode::NOT_IMPLEMENTED:
      return EXECUTE_RESULT::NOT_SUPPORT;
    case StatusCode::INVALID_GRAPH:
      return EXECUTE_RESULT::INVALID_GRAPH;
    default:
      return EXECUTE_RESULT::UNKNOWN_ERROR;
  }
}

EXECUTE_RESULT ExecuteModelWithProtobufs(InferenceSession& sess, const std::vector<onnx::TensorProto>& input_pbs,
                                         const std::vector<onnx::TensorProto>& output_pbs, const char* test_case_name,
                                         const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& input_value_info, Lotus::Test::AllocatorManager& allocatorManager) {
  auto cpu_allocator = allocatorManager.GetAllocator(CPU);
  std::unordered_map<std::string, MLValue> feeds = ConvertPbsToMLValues(input_value_info, input_pbs, cpu_allocator);
  std::vector<MLValue> p_fetches;
  try {
    Common::Status status = sess.Run(feeds, &p_fetches);
    if (!status.IsOK()) {
      LOGF_DEFAULT(ERROR, "%s:%s\n", test_case_name, status.ErrorMessage().c_str());
      return StatusCodeToExecuteResult(status.Code());
    }
  } catch (std::exception& ex) {
    LOGF_DEFAULT(ERROR, "%s:%s\n", test_case_name, ex.what());
    return EXECUTE_RESULT::WITH_EXCEPTION;
  } catch (...) {
    LOGF_DEFAULT(ERROR, "%s:got unknown error\n", test_case_name);
    return EXECUTE_RESULT::WITH_EXCEPTION;
  }

  for (size_t i = 0; i != output_pbs.size(); ++i) {
    const Tensor& outvalue = p_fetches.at(i).Get<Tensor>();
    if (p_fetches.at(i).Fence())
      p_fetches.at(i).Fence()->BeforeUsingAsInput(LotusIR::kCpuExecutionProvider, 0);

    const onnx::TensorProto& expected_value = output_pbs.at(i);
    if (!IsTheSameType(outvalue.DataType(), expected_value.data_type())) {
      LOGF_DEFAULT(ERROR, "%s:type mismatch\n", test_case_name);
      return EXECUTE_RESULT::TYPE_MISMATCH;
    }
    //TODO: support comparisons other than float/bool/int32/...
    auto ret = compare(outvalue, expected_value, cpu_allocator);
    EXECUTE_RESULT compare_result = ret.first;
    if (compare_result == EXECUTE_RESULT::RESULT_DIFFERS)
      LOGF_DEFAULT(ERROR, "%s: the %zd-th value of the %zd-th output differs\n", test_case_name, ret.second, i);
    if (compare_result != EXECUTE_RESULT::SUCCESS)
      return compare_result;
  }
  LOGF_DEFAULT(ERROR, "test %s succeeded\n", test_case_name);
  return EXECUTE_RESULT::SUCCESS;
}

void RunTests(TestEnv& env, int p_models, int concurrent_runs) {
  TestResultStat& stat = env.stat;
  stat.total_test_case_count = std::accumulate(env.tests.begin(), env.tests.end(), (size_t)0, [](size_t v, const TestCaseInfo& info) {
    return info.input_pb_files.size() + v;
  });
  std::vector<TestCaseResult> results(env.tests.size());
#ifdef _WIN32
  if (p_models > 1 && stat.total_test_case_count > 1) {
    ParallelRunTests(env, p_models, concurrent_runs, results);
  } else
#endif
  {
    //run models one by one
    FixedCountFinishCallback c((int)env.tests.size());
    for (size_t i = 0; i != env.tests.size(); ++i) {
      RunSingleTestCase(env, i, concurrent_runs, [i, &results, &c](TestCaseResult& result) {
        results[i] = result;
        c.onFinished(0);
      });
    }
    c.wait();
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

std::vector<TestCaseInfo> LoadTests(const std::vector<string>& input_paths, const std::vector<std::string>& whitelisted_test_cases) {
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

RunContext::RunContext(const TestCaseInfo& test_case1, const std::string& node_name1, std::shared_ptr<Lotus::InferenceSession> session1,
                       const google::protobuf::RepeatedPtrField< ::ONNX_NAMESPACE::ValueInfoProto>& input_info1, Lotus::Test::AllocatorManager& allocatorManager1,
                       std::function<void(TestCaseResult& result)> on_finished1) : test_case(test_case1), node_name(node_name1), session(session1), input_info(input_info1), allocatorManager(allocatorManager1), on_finished(on_finished1), next_test_to_run(0), finished(0), result{std::vector<EXECUTE_RESULT>(test_case1.input_pb_files.size(), EXECUTE_RESULT::UNKNOWN_ERROR), ""} {
}

void RunSingleTestCase(TestEnv& env, size_t test_index, size_t concurrent_runs, std::function<void(TestCaseResult& result)> on_finished) {
  TestCaseResult ret;
  {
    const TestCaseInfo& info = env.tests[test_index];
    const AllocationPlannerType planner = env.planner;
    onnx::ModelProto model_pb;
    {
      std::ifstream input(info.model_url, std::ios::in | std::ios::binary);
      if (!input) {
        LOGF_DEFAULT(ERROR, "open file failed");
        ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), ""};
        goto end;
      }
      if (!model_pb.ParseFromIstream(&input)) {
        LOGF_DEFAULT(ERROR, "parse file failed");
        ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), ""};
        goto end;
      }
    }
    std::string node_name;
    if (model_pb.graph().node().size() == 1) {
      node_name = model_pb.graph().node()[0].op_type();
    }
    SessionOptions so;
    so.allocation_planner_type = planner;
    auto session_object = std::make_shared<InferenceSession>(so);
    Common::Status status;

    if (env.provider == LotusIR::kCudaExecutionProvider) {
#if USE_CUDA
      CUDAExecutionProviderInfo cuda_epi;
      cuda_epi.device_id = 0;
      status = session_object->RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(cuda_epi));
      if (!status.IsOK()) {
        LOGF_DEFAULT(ERROR, "init session %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
        ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), StatusCodeToExecuteResult(status.Code())), node_name};
        goto end;
      }
#else
      LOTUS_THROW("This executable is not built with CUDA");
#endif
    }

    CPUExecutionProviderInfo epi;
    status = session_object->RegisterExecutionProvider(std::make_unique<CPUExecutionProvider>(epi));
    if (!status.IsOK()) {
      LOGF_DEFAULT(ERROR, "init session %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
      ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), StatusCodeToExecuteResult(status.Code())), node_name};
      goto end;
    }

    status = session_object->Load(info.model_url);
    if (!status.IsOK()) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
      ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), StatusCodeToExecuteResult(status.Code())), node_name};
      goto end;
    }
    try {
      status = session_object->Initialize();
      if (!status.IsOK()) {
        LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
        ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), StatusCodeToExecuteResult(status.Code())), node_name};
        goto end;
      }
    } catch (Lotus::NotImplementedException& ex) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info.test_case_name.c_str(), ex.what());
      ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::NOT_SUPPORT), node_name};
      goto end;
    } catch (std::exception& ex) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info.test_case_name.c_str(), ex.what());
      ret = {std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), node_name};
      goto end;
    }
    LOGF_DEFAULT(INFO, "testing %s\n", info.test_case_name.c_str());
#ifdef _WIN32
    if (concurrent_runs > 1) {
      ParallelRunData(std::make_shared<RunContext>(info, node_name, session_object, model_pb.graph().input(), env.allocatorManager, on_finished), concurrent_runs);
      return;
    } else
#endif
    {
      size_t datasets = info.input_pb_files.size();
      ret.node_name = node_name;
      for (size_t i = 0; i != datasets; ++i) {
        std::vector<onnx::TensorProto> input_pbs = LoadTensors(info.input_pb_files[i]);
        std::vector<onnx::TensorProto> output_pbs = LoadTensors(info.output_pb_files[i]);
        ret.excution_result.push_back(ExecuteModelWithProtobufs(*session_object, input_pbs, output_pbs, info.test_case_name.c_str(), model_pb.graph().input(), env.allocatorManager));
      }
      goto end;
    }
  }
end:
  on_finished(ret);
}
