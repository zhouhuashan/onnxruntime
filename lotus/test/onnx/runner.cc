#include "runner.h"
#include <core/common/logging/logging.h>
#include <core/framework/tensorprotoutils.h>

using std::experimental::filesystem::v1::path;
using namespace Lotus;

namespace {
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
      LOTUS_NOT_IMPLEMENTED("Onnx type: ", expected_value.data_type(), " is not supported.");
  }
}

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

EXECUTE_RESULT ExecuteModelWithProtobufs(InferenceSession& sess, const std::vector<onnx::TensorProto>& input_pbs,
                                         const std::vector<onnx::TensorProto>& output_pbs, const char* test_case_name,
                                         const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& input_value_info, AllocatorManager& allocatorManager) {
  auto& cpu_allocator = allocatorManager.GetArena(CPU);
  std::unordered_map<std::string, MLValue> feeds = ConvertPbsToMLValues(input_value_info, input_pbs, cpu_allocator);
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

RunContext::RunContext(const TestCaseInfo& test_case1, const std::string& node_name1, std::shared_ptr<Lotus::InferenceSession> session1,
                       const google::protobuf::RepeatedPtrField< ::ONNX_NAMESPACE::ValueInfoProto>& input_info1, Lotus::AllocatorManager& allocatorManager1,
                       std::function<void(TestCaseResult& result)> on_finished1) : test_case(test_case1), node_name(node_name1), session(session1), input_info(input_info1), allocatorManager(allocatorManager1), on_finished(on_finished1), next_test_to_run(0), finished(0), result{std::vector<EXECUTE_RESULT>(test_case1.input_pb_files.size(), EXECUTE_RESULT::UNKNOWN_ERROR), ""} {
}
void RunSingleTestCase(TestEnv& env, size_t test_index, size_t concurrent_runs, std::function<void(TestCaseResult& result)> on_finished) {
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
  SessionOptions* so = new SessionOptions(vector<ProviderOption>{po});
  so->allocation_planner_type = planner;
  std::shared_ptr<InferenceSession> session_object = std::make_shared<InferenceSession>(*so);
  Common::Status status = session_object->Load(info.model_url);
  if (!status.IsOK()) {
    fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
    TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::LOAD_MODEL_FAILED), node_name};
    on_finished(ret);
    return;
  }
  try {
    status = session_object->Initialize();
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
#ifdef _WIN32
  if (concurrent_runs > 1) {
    RunContext* c = new RunContext(info, node_name, session_object, model_pb.graph().input(), env.allocatorManager, on_finished);
    ParallelRunData(c, concurrent_runs);
  } else
#endif
  {
    size_t datasets = info.input_pb_files.size();
    TestCaseResult ret{std::vector<EXECUTE_RESULT>(datasets, EXECUTE_RESULT::UNKNOWN_ERROR), node_name};
    for (size_t i = 0; i != datasets; ++i) {
      std::vector<onnx::TensorProto> input_pbs = LoadTensors(info.input_pb_files[i]);
      std::vector<onnx::TensorProto> output_pbs = LoadTensors(info.output_pb_files[i]);
      ret.excution_result[i] = ExecuteModelWithProtobufs(*session_object, input_pbs, output_pbs, info.test_case_name.c_str(), model_pb.graph().input(), env.allocatorManager);
    }
    on_finished(ret);
  }
}