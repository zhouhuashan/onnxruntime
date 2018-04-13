#include <core/framework/init.h>
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

using namespace std::experimental::filesystem::v1;
using namespace LotusIR;
using namespace Lotus;

namespace {
//TODO: move it to a common location
MLDataType ElementTypeFromProto(::onnx::TensorProto_DataType type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetType<float>();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetType<bool>();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetType<int>();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetType<double>();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetType<std::string>();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetType<uint8_t>();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetType<uint16_t>();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetType<int16_t>();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetType<int64_t>();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetType<uint32_t>();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetType<uint64_t>();
    default:
      LOTUS_NOT_IMPLEMENTED;
  }
}

inline bool IsTheSameType(MLDataType left, ::onnx::TensorProto_DataType right) {
  return ElementTypeFromProto(right) == left;
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
std::unordered_map<std::string, MLValue> ConvertPbsToMLValues(const ::google::protobuf::RepeatedPtrField< ::onnx::ValueInfoProto>& input_value_info, const std::vector<onnx::TensorProto>& input_pbs, IAllocator& allocator) {
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
        if (use_number_names && std::find_if(input_value_info.begin(), input_value_info.end(), [buf](const ::onnx::ValueInfoProto& info) {
                                  return info.name() == buf;
                                }) == input_value_info.end()) use_number_names = false;
        if (use_data_number_names && std::find_if(input_value_info.begin(), input_value_info.end(), [buf2](const ::onnx::ValueInfoProto& info) {
                                       return info.name() == buf2;
                                     }) == input_value_info.end()) use_data_number_names = false;
        if (use_data_number_names && std::find_if(input_value_info.begin(), input_value_info.end(), [buf3](const ::onnx::ValueInfoProto& info) {
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
          throw std::runtime_error("cannot guess a valid input name");
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

enum class EXECUTE_RESULT {
  SUCCESS = 0,
  UNKNOWN_ERROR = -1,
  FAILED_TO_RUN = -2,
  RESULT_DIFFERS = -3,
  SHAPE_MISMATCH = -4,
  TYPE_MISMATCH = -5,
  NOT_SUPPORT = -6,
};

template <typename FLOAT_TYPE>
EXECUTE_RESULT compare_float_result(const Tensor& outvalue, const Tensor& expected_value, const char* test_case_name, int i) {
  const float abs_error = 1e-6f;
  if (expected_value.Shape() != outvalue.Shape()) return EXECUTE_RESULT::SHAPE_MISMATCH;
  const size_t size1 = expected_value.Shape().Size();
  const FLOAT_TYPE* expected_output = expected_value.Data<FLOAT_TYPE>();
  const FLOAT_TYPE* real_output = outvalue.Data<FLOAT_TYPE>();
  for (size_t di = 0; di != size1; ++di) {
    const double diff = fabs(expected_output[di] - real_output[di]);
    if (diff > abs_error) {
      fprintf(stderr, "%s: the %d-th value of the %d-th output differs\n", test_case_name, (int)di, (int)i);
      return EXECUTE_RESULT::RESULT_DIFFERS;
    }
  }
  return EXECUTE_RESULT::SUCCESS;
}

template <typename T>
EXECUTE_RESULT is_result_exactly_match(const Tensor& outvalue, const Tensor& expected_value, const char* test_case_name, int i) {
  if (expected_value.Shape() != outvalue.Shape()) return EXECUTE_RESULT::SHAPE_MISMATCH;
  const size_t size1 = expected_value.Shape().Size();
  const T* expected_output = expected_value.Data<T>();
  const T* real_output = outvalue.Data<T>();
  for (size_t di = 0; di != size1; ++di) {
    if (expected_output[di] != real_output[di]) {
      fprintf(stderr, "%s: the %d-th value of the %d-th output differs\n", test_case_name, (int)di, (int)i);
      return EXECUTE_RESULT::RESULT_DIFFERS;
    }
  }
  return EXECUTE_RESULT::SUCCESS;
}

EXECUTE_RESULT ExecuteModelWithProtobufs(InferenceSession& sess, const std::vector<onnx::TensorProto>& input_pbs,
                                         const std::vector<onnx::TensorProto>& output_pbs, const char* test_case_name,
                                         const onnx::ModelProto& model_pb) {
  auto& cpu_allocator = AllocatorManager::Instance().GetArena(CPU);
  for (const ValueInfoProto& a : model_pb.graph().input()) {
    if (a.type().tensor_type().elem_type() != TensorProto_DataType_FLOAT)
      return EXECUTE_RESULT::NOT_SUPPORT;
  }
  for (const ValueInfoProto& a : model_pb.graph().output()) {
    if (a.type().tensor_type().elem_type() != TensorProto_DataType_FLOAT)
      return EXECUTE_RESULT::NOT_SUPPORT;
  }
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
    std::unique_ptr<Tensor> expected_tensor;
    LOTUS_ENFORCE(Lotus::Utils::GetTensorFromTensorProto(expected_value, &expected_tensor, cpu_allocator).IsOK());
    //TODO: support comparisons other than float/bool/int32/...
    EXECUTE_RESULT compare_result = EXECUTE_RESULT::UNKNOWN_ERROR;
    switch (expected_value.data_type()) {
      case TensorProto_DataType_FLOAT:
        compare_result = compare_float_result<float>(outvalue, *expected_tensor.get(), test_case_name, (int)i);
        break;
      case TensorProto_DataType_DOUBLE:
        compare_result = compare_float_result<double>(outvalue, *expected_tensor.get(), test_case_name, (int)i);
        break;
      case TensorProto_DataType_BOOL:
        compare_result = is_result_exactly_match<bool>(outvalue, *expected_tensor.get(), test_case_name, (int)i);
        break;
      case TensorProto_DataType_INT32:
        compare_result = is_result_exactly_match<int32_t>(outvalue, *expected_tensor.get(), test_case_name, (int)i);
        break;
      case TensorProto_DataType_INT64:
        compare_result = is_result_exactly_match<int64_t>(outvalue, *expected_tensor.get(), test_case_name, (int)i);
        break;
      default:
        LOTUS_NOT_IMPLEMENTED;
    }
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
      throw std::runtime_error("open file failed");
    }
    if (!tensor.ParseFromIstream(&input)) {
      throw std::runtime_error("parse file failed");
    }
    input_pbs.emplace_back(tensor);
  }
  return input_pbs;
}

void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-m TEST_MODE: TEST_MODE could be 'node' or 'model'. Default: 'node'.\n"
      "\t-p PLANNER_TYPE: PLANNER_TYPE could be 'seq' or 'simple'. Default: 'simple'.\n"
      "\t-h: help\n");
  exit(-1);
}

template <typename T1>
std::string containerToStr(const T1& input) {
  std::ostringstream oss;
  bool is_first = true;
  std::vector<typename T1::value_type> vec(input.begin(), input.end());
  std::sort(vec.begin(), vec.end());
  for (const auto& s : vec) {
    if (!is_first) oss << ", ";
    oss << s;
    is_first = false;
  }
  return oss.str();
}

struct TestCaseInfo {
  std::string model_url;
  std::string test_case_name;
  std::vector<path> input_pb_files;
  std::vector<path> output_pb_files;
};

struct NodeTestResultStat {
  int total_test_case_count = 0;
  int succeeded = 0;
  int not_implemented = 0;
  int load_model_failed = 0;
  int throwed_exception = 0;
  int result_differs = 0;
  int skipped = 0;
  std::unordered_set<std::string> not_implemented_kernels;
  std::unordered_set<std::string> failed_kernels;
  std::unordered_set<std::string> covered_ops;
};

struct ModelTestResultStat {
  int total_test_case_count = 0;
  int succeeded = 0;
  int not_implemented = 0;
  int load_model_failed = 0;
  int throwed_exception = 0;
  int result_differs = 0;
  std::unordered_set<std::string> not_implemented_kernels;
  std::unordered_set<std::string> covered_ops;
};

void print_result(const ModelTestResultStat& stat) {
  std::string not_implemented_kernels_str = containerToStr(stat.not_implemented_kernels);
  int failed = stat.total_test_case_count - stat.succeeded;
  int other_reason_failed = failed - stat.not_implemented - stat.load_model_failed - stat.result_differs - stat.throwed_exception;
  printf(
      "result: \n"
      "\tTotal test cases:%d\n"
      "\t\tSucceeded:%d\n"
      "\t\tFailed:%d\n"
      "\t\t\tKernel not implemented:%d\n"
      "\t\t\tLoad model Failed:%d\n"
      "\t\t\tThrew exception while runnning:%d\n"
      "\t\t\tResult differs:%d\n"
      "\t\t\tOther reason:%d\n"
      "\tTotal OPs covered:%d\n",
      stat.total_test_case_count, stat.succeeded, failed, stat.not_implemented, stat.load_model_failed, stat.throwed_exception, stat.result_differs, other_reason_failed, (int)stat.covered_ops.size());
}
void print_result(const NodeTestResultStat& stat, const std::vector<std::string>& all_implemented_ops, bool no_coverage_info) {
  std::unordered_set<std::string> succeeded_kernels(stat.covered_ops);
  std::vector<std::string> not_tested;
  for (const std::string& s : all_implemented_ops) {
    if (stat.covered_ops.find(s) == stat.covered_ops.end())
      not_tested.push_back(s);
  }
  for (const std::string& name : stat.not_implemented_kernels) {
    succeeded_kernels.erase(name);
  }
  for (const std::string& name : stat.failed_kernels) {
    succeeded_kernels.erase(name);
  }
  std::string not_implemented_kernels_str = containerToStr(stat.not_implemented_kernels);
  std::string failed_kernels_str = containerToStr(stat.failed_kernels);
  std::string not_tested_str = containerToStr(not_tested);
  int failed = stat.total_test_case_count - stat.succeeded;
  int other_reason_failed = failed - stat.not_implemented - stat.load_model_failed - stat.result_differs - stat.throwed_exception - stat.skipped;
  std::ostringstream oss;
  oss << "result: \n"
         "\tTotal test cases:"
      << stat.total_test_case_count
      << "\n\t\tSucceeded:" << stat.succeeded
      << "\n\t\tSkipped:" << stat.skipped
      << "\n\t\tFailed:" << failed << "\n\t\t\tKernel not implemented:"
      << stat.not_implemented << "\n\t\t\tLoad model Failed:"
      << stat.load_model_failed << "\n\t\t\tThrew exception while runnning:"
      << stat.throwed_exception << "\n\t\t\tResult_differs:"
      << stat.result_differs << "\n\t\t\tOther reason:"
      << other_reason_failed << "\n";
  if (!no_coverage_info) {
    oss << "\tTotal OPs implemented:" << all_implemented_ops.size() << "\n\t\tNot covered by any test("
        << not_tested.size() << "): " << not_tested_str << "\n\tTotal OPs covered:"
        << stat.covered_ops.size() << "\n\t\tSucceeded:"
        << succeeded_kernels.size() << "\n";
  }
  oss << "\t\tNot implemented Kernels(" << stat.not_implemented_kernels.size() << "): " << not_implemented_kernels_str << "\n\tFailed Kernels:"
      << failed_kernels_str << "\n";
  std::string res = oss.str();
  fwrite(res.c_str(), 1, res.size(), stdout);
}

void RunNodeTests(const std::vector<TestCaseInfo>& tests, const std::vector<std::string>& all_implemented_ops, NodeTestResultStat& stat, AllocationPlannerType planner) {
  for (TestCaseInfo info : tests) {
    onnx::ModelProto model_pb;
    {
      std::ifstream input(info.model_url, std::ios::in | std::ios::binary);
      if (!input) {
        fprintf(stderr, "open file failed");
        continue;
      }
      if (!model_pb.ParseFromIstream(&input)) {
        fprintf(stderr, "parse file failed");
        continue;
      }
    }
    if (model_pb.graph().node().size() != 1) {
      fprintf(stderr, "not a node test");
      continue;
    }
    ++stat.total_test_case_count;
    auto& node_pb = model_pb.graph().node()[0];
    stat.covered_ops.insert(node_pb.op_type());
    if (std::find(all_implemented_ops.begin(), all_implemented_ops.end(), node_pb.op_type()) == all_implemented_ops.end()) {
      stat.not_implemented_kernels.insert(node_pb.op_type());
      ++stat.not_implemented;
      continue;
    }
    ExecutionProviderInfo epi;
    ProviderOption po{"CPUExecutionProvider", epi};
    SessionOptions so(vector<ProviderOption>{po});
    so.allocation_planner_type = planner;
    InferenceSession session_object{so};
    Common::Status status = session_object.Load(info.model_url);
    if (!status.IsOK()) {
      fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
      ++stat.load_model_failed;
      stat.failed_kernels.insert(node_pb.op_type());
      continue;
    }
    try {
      status = session_object.Initialize();
      if (!status.IsOK()) {
        fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
        ++stat.load_model_failed;
        stat.failed_kernels.insert(node_pb.op_type());
        continue;
      }
    } catch (std::exception& ex) {
      fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), ex.what());
      ++stat.load_model_failed;
      stat.failed_kernels.insert(node_pb.op_type());
      continue;
    }
    fprintf(stderr, "testing %s\n", info.test_case_name.c_str());
    std::vector<onnx::TensorProto> input_pbs = LoadTensors(info.input_pb_files);
    std::vector<onnx::TensorProto> output_pbs = LoadTensors(info.output_pb_files);
    EXECUTE_RESULT ret = ExecuteModelWithProtobufs(session_object, input_pbs, output_pbs, info.test_case_name.c_str(), model_pb);
    if (ret == EXECUTE_RESULT::SUCCESS) {
      ++stat.succeeded;
    } else if (ret == EXECUTE_RESULT::FAILED_TO_RUN) {
      ++stat.throwed_exception;
      stat.failed_kernels.insert(node_pb.op_type());
    } else if (ret == EXECUTE_RESULT::RESULT_DIFFERS) {
      ++stat.result_differs;
      stat.failed_kernels.insert(node_pb.op_type());
    } else if (ret == EXECUTE_RESULT::NOT_SUPPORT) {
      ++stat.skipped;
    } else {
      stat.failed_kernels.insert(node_pb.op_type());
    }
  }
}

void RunModelTests(const std::vector<TestCaseInfo>& tests, const std::vector<std::string>& all_implemented_ops, ModelTestResultStat& stat, AllocationPlannerType planner) {
  for (TestCaseInfo info : tests) {
    onnx::ModelProto model_pb;
    {
      std::ifstream input(info.model_url, std::ios::in | std::ios::binary);
      if (!input) {
        fprintf(stderr, "open file failed");
        continue;
      }
      if (!model_pb.ParseFromIstream(&input)) {
        fprintf(stderr, "parse file failed");
        continue;
      }
    }
    ++stat.total_test_case_count;
    bool can_be_loaded = true;
    for (const auto& node_pb : model_pb.graph().node()) {
      stat.covered_ops.insert(node_pb.op_type());
      if (std::find(all_implemented_ops.begin(), all_implemented_ops.end(), node_pb.op_type()) == all_implemented_ops.end()) {
        stat.not_implemented_kernels.insert(node_pb.op_type());
        can_be_loaded = false;
        continue;
      }
    }
    if (!can_be_loaded) {
      ++stat.not_implemented;
      continue;
    }
    ExecutionProviderInfo epi;
    ProviderOption po{"CPUExecutionProvider", epi};
    SessionOptions so(vector<ProviderOption>{po});
    so.allocation_planner_type = planner;
    InferenceSession session_object{so};
    Common::Status status = session_object.Load(info.model_url);
    if (!status.IsOK()) {
      fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
      ++stat.load_model_failed;
      continue;
    }
    try {
      status = session_object.Initialize();
      if (!status.IsOK()) {
        fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), status.ErrorMessage().c_str());
        ++stat.load_model_failed;
        continue;
      }
    } catch (std::exception& ex) {
      fprintf(stderr, "load model %s failed:%s\n", info.test_case_name.c_str(), ex.what());
      ++stat.load_model_failed;
      continue;
    }
    fprintf(stderr, "testing %s\n", info.test_case_name.c_str());
    std::vector<onnx::TensorProto> input_pbs = LoadTensors(info.input_pb_files);
    std::vector<onnx::TensorProto> output_pbs = LoadTensors(info.output_pb_files);
    EXECUTE_RESULT ret = ExecuteModelWithProtobufs(session_object, input_pbs, output_pbs, info.test_case_name.c_str(), model_pb);
    if (ret == EXECUTE_RESULT::SUCCESS) {
      ++stat.succeeded;
    } else if (ret == EXECUTE_RESULT::RESULT_DIFFERS) {
      ++stat.result_differs;
    } else {
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
    throw std::runtime_error("parse file name failed");
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
      throw std::runtime_error("illegal input file names");
    }
  }
}

std::vector<TestCaseInfo> GatherTests(const std::string& test_case_name, const path& test_case_dir) {
  std::string model_file_path = (test_case_dir / "model.onnx").string();
  path pb(".pb");
  std::vector<TestCaseInfo> ret;
  for (directory_iterator test_data_set(test_case_dir), end2; test_data_set != end2; ++test_data_set) {
    if (!is_directory(*test_data_set)) {
      continue;
    }
    TestCaseInfo info;
    info.test_case_name = test_case_name;
    info.model_url = model_file_path;
    for (directory_iterator pb_file(*test_data_set), end3; pb_file != end3; ++pb_file) {
      path f = *pb_file;
      if (!is_regular_file(f)) continue;
      if (f.extension() != pb) continue;
      std::string filename = f.filename().string();
      if (!filename.compare(0, 6, "input_")) {
        info.input_pb_files.push_back(f);
      } else if (!filename.compare(0, 7, "output_")) {
        info.output_pb_files.push_back(f);
      }
    }
    SortTensorFileNames(info.input_pb_files);
    SortTensorFileNames(info.output_pb_files);
    ret.push_back(info);
  }
  return ret;
}
std::vector<TestCaseInfo> LoadTests(const path& test_data_root_path, const std::vector<std::string>& whitelisted_test_cases) {
  std::vector<TestCaseInfo> tests;
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
    std::vector<TestCaseInfo> vec = GatherTests(test_case_name, test_case_dir->path());
    tests.insert(tests.end(), vec.begin(), vec.end());
  }
  return tests;
}

enum class RUN_MODE {
  NODE_TEST,
  MODEL_TEST,
};
}  // namespace

int main(int argc, char* argv[]) {
  std::string default_logger_id{"Default"};
  Logging::LoggingManager default_logging_manager{std::unique_ptr<Logging::ISink>{new Logging::CLogSink{}},
                                                  Logging::Severity::kWARNING, false, Logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};
  Initializer::EnsureInitialized(&argc, &argv);
  int ch;
  RUN_MODE mode = RUN_MODE::NODE_TEST;
  AllocationPlannerType planner = AllocationPlannerType::SIMPLE_SEQUENTIAL_PLANNER;
  //if this var is not empty, only run the tests with name in this list
  std::vector<std::string> whitelisted_test_cases;
  while ((ch = getopt(argc, argv, "hm:n:p:")) != -1) {
    switch (ch) {
      case 'm':
        if (!strcmp(optarg, "node")) {
          mode = RUN_MODE::NODE_TEST;
        } else if (!strcmp(optarg, "model")) {
          mode = RUN_MODE::MODEL_TEST;
        } else {
          usage();
          return -1;
        }
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
  argc -= optind;
  argv += optind;
  if (argc < 1) {
    fprintf(stderr, "please specify dataroot\n");
    usage();
    return -1;
  }
  std::vector<std::string> all_implemented_ops = KernelRegistry::Instance().GetAllRegisteredOpNames();
  const char* test_data_root = argv[0];
  path test_data_root_path(test_data_root);
  if (mode == RUN_MODE::NODE_TEST) {
    NodeTestResultStat stat;
    std::vector<TestCaseInfo> tests = LoadTests(test_data_root_path, whitelisted_test_cases);
    RunNodeTests(tests, all_implemented_ops, stat, planner);
    print_result(stat, all_implemented_ops, !whitelisted_test_cases.empty());
  } else {
    ModelTestResultStat stat;
    std::vector<TestCaseInfo> tests = LoadTests(test_data_root_path, whitelisted_test_cases);
    RunModelTests(tests, all_implemented_ops, stat, planner);
    print_result(stat);
  }

  return 0;
}
