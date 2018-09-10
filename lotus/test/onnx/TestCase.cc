#include "TestCase.h"
#include <fstream>
#include <memory>

#include "core/platform/env.h"
#include "core/framework/tensorprotoutils.h"
#include <google/protobuf/util/delimited_message_util.h>
#include <google/protobuf/text_format.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4018) /*'expression' : signed/unsigned mismatch */
#pragma warning(disable : 4065) /*switch statement contains 'default' but no 'case' labels*/
#pragma warning(disable : 4100)
#pragma warning(disable : 4146) /*unary minus operator applied to unsigned type, result still unsigned*/
#pragma warning(disable : 4244) /*'conversion' conversion from 'type1' to 'type2', possible loss of data*/
#pragma warning(disable : 4251) /*'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'*/
#pragma warning(disable : 4267) /*'var' : conversion from 'size_t' to 'type', possible loss of data*/
#pragma warning(disable : 4305) /*'identifier' : truncation from 'type1' to 'type2'*/
#pragma warning(disable : 4307) /*'operator' : integral constant overflow*/
#pragma warning(disable : 4309) /*'conversion' : truncation of constant value*/
#pragma warning(disable : 4334) /*'operator' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)*/
#pragma warning(disable : 4355) /*'this' : used in base member initializer list*/
#pragma warning(disable : 4506) /*no definition for inline function 'function'*/
#pragma warning(disable : 4800) /*'type' : forcing value to bool 'true' or 'false' (performance warning)*/
#pragma warning(disable : 4996) /*The compiler encountered a deprecated declaration.*/
#endif
#include "tml.pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

using namespace std::experimental::filesystem::v1;

using namespace onnxruntime;
using namespace onnxruntime::common;

namespace {
template <typename InputType, typename OutputType>
Status ConvertVector(const InputType& data, OutputType** vec) {
  //void* p = allocator->Alloc(sizeof(OutputType));
  //if (p == nullptr)
  //	return Status(LOTUS, FAIL, "out of memory");
  //OutputType* v = new (p) OutputType();
  //TODO: non-tensor type has no deleter inside it. So, cannot use allocator
  OutputType* v = new OutputType();
  for (const auto& i : data) {
    typename OutputType::value_type new_value;
    for (const auto& j : i.v()) {
      new_value[j.first] = j.second;
    }
    v->push_back(new_value);
  }
  *vec = v;
  return Status::OK();
}

template <typename InputType, typename OutputType>
Status Convert(const InputType& tensor_proto, OutputType** p_tensor);

template <>
Status Convert(const google::protobuf::RepeatedPtrField<proto::MapInt64ToFloat>& data, VectorMapInt64ToFloat** vec) {
  return ConvertVector<google::protobuf::RepeatedPtrField<proto::MapInt64ToFloat>, VectorMapInt64ToFloat>(data, vec);
}

template <>
Status Convert(const google::protobuf::RepeatedPtrField<proto::MapStringToFloat>& data, VectorMapStringToFloat** vec) {
  return ConvertVector<google::protobuf::RepeatedPtrField<proto::MapStringToFloat>, VectorMapStringToFloat>(data, vec);
}

template <typename InputType, typename OutputType>
void ConvertMap(const InputType& data, OutputType** out) {
  OutputType* ret = new OutputType();
  for (const auto& pv : data) {
    (*ret)[pv.first] = pv.second;
  }
  *out = ret;
}

template <>
Status Convert(const google::protobuf::Map<std::string, std::string>& data, MapStringToString** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<std::string, int64_t>& data, MapStringToInt64** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<std::string, float>& data, MapStringToFloat** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<std::string, double>& data, MapStringToDouble** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, std::string>& data, MapInt64ToString** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, int64_t>& data, MapInt64ToInt64** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, float>& data, MapInt64ToFloat** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, double>& data, MapInt64ToDouble** out) {
  ConvertMap(data, out);
  return Status::OK();
}
template <typename InputType, typename OutputType>
Status RichTypeProtoToMLValue(const InputType& input, MLValue& value) {
  OutputType* tensor = nullptr;
  Status st = Convert(input, &tensor);
  if (!st.IsOK()) return st;
  value.Init(tensor,
             DataTypeImpl::GetType<OutputType>(),
             DataTypeImpl::GetType<OutputType>()->GetDeleteFunc());
  return Status::OK();
}

static int ExtractFileNo(const std::string& name) {
  size_t p1 = name.rfind('.');
  size_t p2 = name.rfind('_', p1);
  ++p2;
  std::string number_str = name.substr(p2, p1 - p2);
  const char* start = number_str.c_str();
  const char* end = number_str.c_str();
  long ret = strtol(start, const_cast<char**>(&end), 10);
  if (end == start) {
    LOTUS_THROW("parse file name failed");
  }
  return static_cast<int>(ret);
}

static Status SortTensorFileNames(std::vector<path>& input_pb_files) {
  if (input_pb_files.size() <= 1) return Status::OK();
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
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "illegal input file name:", input_pb_files[i].filename().string());
    }
  }
  return Status::OK();
}

//Doesn't support file size >2 GB
template <typename FUNC>
Status LoopDataFile(const path& outputs_pb, AllocatorPtr allocator, FUNC func) {
  std::string content;
  //TODO: mmap is better
  LOTUS_RETURN_IF_ERROR(Env::Default().ReadFileAsString(outputs_pb.c_str(), &content));
  google::protobuf::io::CodedInputStream coded_input((const uint8_t*)content.data(), (int)content.size());
  bool clean_eof = false;
  Status st;
  int item_id = 1;
  for (proto::TraditionalMLData data; google::protobuf::util::ParseDelimitedFromCodedStream(&data, &coded_input, &clean_eof); ++item_id, data.Clear()) {
    MLValue value;
    switch (data.values_case()) {
      case proto::TraditionalMLData::kVectorMapStringToFloat:
        st = RichTypeProtoToMLValue<decltype(data.vector_map_string_to_float().v()), VectorMapStringToFloat>(data.vector_map_string_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kVectorMapInt64ToFloat:
        st = RichTypeProtoToMLValue<decltype(data.vector_map_int64_to_float().v()), VectorMapInt64ToFloat>(data.vector_map_int64_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToString:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_string().v()), MapStringToString>(data.map_string_to_string().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToInt64:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_int64().v()), MapStringToInt64>(data.map_string_to_int64().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToFloat:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_float().v()), MapStringToFloat>(data.map_string_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToDouble:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_double().v()), MapStringToDouble>(data.map_string_to_double().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToString:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_string().v()), MapInt64ToString>(data.map_int64_to_string().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToInt64:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_int64().v()), MapInt64ToInt64>(data.map_int64_to_int64().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToFloat:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_float().v()), MapInt64ToFloat>(data.map_int64_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToDouble:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_double().v()), MapInt64ToDouble>(data.map_int64_to_double().v(), value);
        break;
      case proto::TraditionalMLData::kTensor:
        st = Utils::TensorProtoToMLValue(data.tensor(), allocator, nullptr, 0, value);
        break;
      default:
        st = Status(LOTUS, NOT_IMPLEMENTED, "unknown data type inside TraditionalMLData");
    }
    if (!st.IsOK()) break;
    st = func(data.name(), &value, data.debug_info());
    if (!st.IsOK()) break;
  }
  if (!st.IsOK()) return LOTUS_MAKE_STATUS(LOTUS, FAIL, "load the ", item_id, "-th item in file '", outputs_pb.string(), "' failed,", st.ErrorMessage());
  if (!clean_eof) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "parse input file '", outputs_pb.string(), "' failed, clean_eof==false");
  }
  return Status::OK();
}

Status loadModel(std::istream& model_istream, onnx::ModelProto* p_model_proto) {
  if (!model_istream.good()) {
    return Status(LOTUS, INVALID_ARGUMENT, "Invalid istream object.");
  }
  if (!p_model_proto) {
    return Status(LOTUS, INVALID_ARGUMENT, "Null model_proto ptr.");
  }
  const bool result = p_model_proto->ParseFromIstream(&model_istream);
  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Failed to load model because protobuf parsing failed.");
  }
  return Status::OK();
}

Status loadModelFile(const std::string& model_url, onnx::ModelProto* model_pb) {
  std::ifstream input(model_url, std::ios::in | std::ios::binary);
  if (!input) {
    std::ostringstream oss;
    oss << "open file " << model_url << " failed";
    return Status(LOTUS, NO_SUCHFILE, oss.str());
  }
  return loadModel(input, model_pb);
}

template <typename T>
static void RepeatedPtrFieldToVector(const ::google::protobuf::RepeatedPtrField<T>& input_value_info, std::vector<T>& out) {
  for (int i = 0; i != input_value_info.size(); ++i) {
    out.push_back(input_value_info[i]);
  }
}
}  // namespace

/**
   * test_case_dir must have contents of:
   * model.onnx
   * ???/input_??.pb
   * ???/output_??.pb
   * ???/input_??.pb
   * ???/output_??.pb
   */
class OnnxTestCase : public ITestCase {
 private:
  std::string test_case_name_;
  std::experimental::filesystem::v1::path model_url_;
  AllocatorPtr allocator_;
  std::vector<std::string> debuginfo_strings;
  std::mutex m_;
  std::vector<onnx::ValueInfoProto> input_value_info_;
  std::vector<onnx::ValueInfoProto> output_value_info_;

  Status FromPbFiles(const std::vector<std::experimental::filesystem::v1::path>& files, std::vector<MLValue>& output_values);
  std::vector<std::experimental::filesystem::v1::path> test_data_dirs_;

  std::string GetDatasetDebugInfoString(size_t dataset_id) override {
    std::lock_guard<std::mutex> l(m_);
    if (dataset_id < debuginfo_strings.size())
      return debuginfo_strings[dataset_id];
    return test_data_dirs_.at(dataset_id).string();
  }
  //If we cannot get input name from input_pbs, we'll use names like "data_0","data_1",... It's dirty hack
  // for https://github.com/onnx/onnx/issues/679
  Status ConvertInput(const std::vector<onnx::TensorProto>& input_pbs, std::unordered_map<std::string, MLValue>& out);
  std::string node_name_;
  std::once_flag model_parsed_;
  std::once_flag config_parsed_;
  double per_sample_tolerance_;
  double relative_per_sample_tolerance_;
  bool post_processing_;
  Status ParseModel();
  Status ParseConfig();
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(OnnxTestCase);

 public:
  OnnxTestCase(const AllocatorPtr&, const std::string& test_case_name);
  explicit OnnxTestCase(const std::string& test_case_name) : test_case_name_(test_case_name) {}
  void SetAllocator(const AllocatorPtr&) override;
  Status GetPerSampleTolerance(double* value) override;
  Status GetRelativePerSampleTolerance(double* value) override;
  Status GetPostProcessing(bool* value) override;

  const onnx::ValueInfoProto& GetOutputInfoFromModel(size_t i) const override {
    return output_value_info_[i];
  }
  size_t GetDataCount() const override {
    return test_data_dirs_.size();
  }
  Status GetNodeName(std::string* out) override {
    Status st = ParseModel();
    if (st.IsOK()) *out = node_name_;
    return st;
  }
  Status SetModelPath(const std::experimental::filesystem::v1::path& path) override;

  const std::experimental::filesystem::v1::path& GetModelUrl() const override {
    return model_url_;
  }
  const std::string& GetTestCaseName() const override {
    return test_case_name_;
  }
  Status LoadInputData(size_t id, std::unordered_map<std::string, MLValue>& feeds) override;
  Status LoadOutputData(size_t id, std::vector<MLValue>& output_values) override;
};

ITestCase* CreateOnnxTestCase(const AllocatorPtr& ptr, const std::string& test_case_name) {
  return new OnnxTestCase(ptr, test_case_name);
}
ITestCase* CreateOnnxTestCase(const std::string& test_case_name) {
  return new OnnxTestCase(test_case_name);
}

Status OnnxTestCase::GetPerSampleTolerance(double* value) {
  Status st = ParseConfig();
  if (!st.IsOK())
    return LOTUS_MAKE_STATUS(LOTUS, MODEL_LOADED, "parse test config failed:", st.ErrorMessage());

  *value = per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetRelativePerSampleTolerance(double* value) {
  Status st = ParseConfig();
  if (!st.IsOK())
    return LOTUS_MAKE_STATUS(LOTUS, MODEL_LOADED, "parse test config failed:", st.ErrorMessage());
  *value = relative_per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetPostProcessing(bool* value) {
  Status st = ParseConfig();
  if (!st.IsOK()) {
    return LOTUS_MAKE_STATUS(LOTUS, MODEL_LOADED, "parse test config failed:", st.ErrorMessage());
  }
  *value = post_processing_;
  return Status::OK();
}

Status OnnxTestCase::ParseConfig() {
  Status st = Status::OK();
  std::call_once(config_parsed_, [this, &st]() {
    path config_path = model_url_.replace_filename("config.txt");
    st = Env::Default().FileExists(config_path.c_str());
    if (!st.IsOK()) {
      per_sample_tolerance_ = 1e-3;
      relative_per_sample_tolerance_ = 1e-5;
      post_processing_ = false;
      st = Status::OK();
      return;
    }
    //parse model
    onnxruntime::proto::TestCaseConfig config_pb;
    std::string body;
    st = Env::Default().ReadFileAsString(config_path.c_str(), &body);
    if (!st.IsOK()) {
      return;
    }
    if (!google::protobuf::TextFormat::ParseFromString(body, &config_pb)) {
      st = Status(LOTUS, FAIL, "Parse config failed");
      return;
    }
    per_sample_tolerance_ = config_pb.per_sample_tolerance();
    relative_per_sample_tolerance_ = config_pb.relative_per_sample_tolerance();
    post_processing_ = config_pb.post_processing();
    st = Status::OK();
  });
  return st;
}
Status OnnxTestCase::ParseModel() {
  Status st = Status::OK();
  std::call_once(model_parsed_, [this, &st]() {
    //parse model
    onnx::ModelProto model_pb;
    st = loadModelFile(model_url_.string(), &model_pb);
    if (!st.IsOK()) return;
    const onnx::GraphProto& graph = model_pb.graph();
    if (graph.node().size() == 1) {
      node_name_ = graph.node()[0].op_type();
    }
    RepeatedPtrFieldToVector(graph.input(), input_value_info_);
    RepeatedPtrFieldToVector(graph.output(), output_value_info_);
    st = Status::OK();
  });
  return st;
}
Status OnnxTestCase::SetModelPath(const path& m) {
  model_url_ = m;
  path test_case_dir = m.parent_path();
  for (directory_iterator test_data_set(test_case_dir), end2; test_data_set != end2; ++test_data_set) {
    if (!is_directory(*test_data_set)) {
      continue;
    }
    test_data_dirs_.push_back(test_data_set->path());
    debuginfo_strings.push_back(test_data_set->path().string());
  }
  return Status::OK();
}

//load tensors from disk
static Status LoadTensors(const std::vector<path>& pb_files, std::vector<onnx::TensorProto>* input_pbs) {
  for (size_t i = 0; i != pb_files.size(); ++i) {
    onnx::TensorProto tensor;
    std::ifstream input(pb_files.at(i), std::ios::in | std::ios::binary);
    if (!input) {
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "open file '", pb_files.at(i), "' failed");
    }
    if (!tensor.ParseFromIstream(&input)) {
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "parse file '", pb_files.at(i), "' failed");
    }
    input_pbs->emplace_back(tensor);
  }
  return Status::OK();
}

Status OnnxTestCase::LoadInputData(size_t id, std::unordered_map<std::string, MLValue>& feeds) {
  if (id >= test_data_dirs_.size())
    return Status(LOTUS, INVALID_ARGUMENT, "out of bound");

  Status st = ParseModel();
  if (!st.IsOK())
    return LOTUS_MAKE_STATUS(LOTUS, MODEL_LOADED, "parse model failed:", st.ErrorMessage());

  path inputs_pb = test_data_dirs_[id] / "inputs.pb";
  if (std::experimental::filesystem::exists(inputs_pb)) {  //has an all-in-one input file
    std::ostringstream oss;
    oss << debuginfo_strings[id];
    st = LoopDataFile(inputs_pb, allocator_, [&feeds, &oss](const std::string& name, MLValue* value, const std::string& debug_info) {
      if (!debug_info.empty()) {
        oss << ":" << debug_info;
      }
      if (name.empty())
        return Status(LOTUS, FAIL, "name is empty");
      auto pv = feeds.insert(std::make_pair(name, *value));
      if (!pv.second)
        return Status(LOTUS, FAIL, "duplicated input name");
      return Status::OK();
    });
    {
      std::lock_guard<std::mutex> l(m_);
      debuginfo_strings[id] = oss.str();
    }
    return st;
  }

  std::vector<path> input_pb_files;
  const path pb(".pb");

  for (directory_iterator pb_file(test_data_dirs_[id]), end3; pb_file != end3; ++pb_file) {
    path f = *pb_file;
    if (!is_regular_file(f)) continue;
    if (f.extension() != pb) continue;
    std::string filename = f.filename().string();
    if (!filename.compare(0, 6, "input_")) {
      input_pb_files.push_back(f);
    }
  }
  LOTUS_RETURN_IF_ERROR(SortTensorFileNames(input_pb_files));

  std::vector<onnx::TensorProto> input_pbs;
  LOTUS_RETURN_IF_ERROR(LoadTensors(input_pb_files, &input_pbs));
  LOTUS_RETURN_IF_ERROR(ConvertInput(input_pbs, feeds));
  return Status::OK();
}

Status OnnxTestCase::FromPbFiles(const std::vector<path>& files, std::vector<MLValue>& output_values) {
  for (const path& f : files) {
    if (!f.has_extension()) return LOTUS_MAKE_STATUS(LOTUS, NOT_IMPLEMENTED, "unknown file type, path = ", f);
    std::string s = f.extension().string();
    if (s != ".pb")
      continue;
    onnx::TensorProto tensor;
    {
      std::ifstream input(f, std::ios::in | std::ios::binary);
      if (!input) {
        return Status(LOTUS, FAIL, "open file failed");
      }
      if (!tensor.ParseFromIstream(&input)) {
        return Status(LOTUS, FAIL, "parse file failed");
      }
    }
    MLValue value;
    LOTUS_RETURN_IF_ERROR(onnxruntime::Utils::TensorProtoToMLValue(tensor, allocator_, nullptr, 0, value));
    output_values.emplace_back(value);
  }
  return Status::OK();
}

Status OnnxTestCase::LoadOutputData(size_t id, std::vector<MLValue>& output_values) {
  if (id >= test_data_dirs_.size())
    return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, test_case_name_, ":Attempt to load output data from directory id of ", id, ". Num data dirs :", test_data_dirs_.size());
  Status st = ParseModel();
  if (!st.IsOK())
    return LOTUS_MAKE_STATUS(LOTUS, MODEL_LOADED, "parse model failed:", st.ErrorMessage());
  path outputs_pb = test_data_dirs_[id] / "outputs.pb";
  output_values.clear();
  if (std::experimental::filesystem::exists(outputs_pb)) {  //has an all-in-one output file
    return LoopDataFile(outputs_pb, allocator_, [&output_values](const std::string&, MLValue* value, const std::string&) {
      output_values.push_back(*value);
      return Status::OK();
    });
  }
  std::vector<path> output_pb_files;
  const path pb(".pb");
  const path tpb(".tpb");
  for (directory_iterator pb_file(test_data_dirs_[id]), end3; pb_file != end3; ++pb_file) {
    path f = *pb_file;
    if (!is_regular_file(f)) continue;
    if (f.extension() != pb && f.extension() != tpb) continue;
    std::string filename = f.filename().string();
    if (!filename.compare(0, 7, "output_")) {
      output_pb_files.push_back(f);
    }
  }
  SortTensorFileNames(output_pb_files);
  LOTUS_RETURN_IF_ERROR(FromPbFiles(output_pb_files, output_values));
  return Status::OK();
}

Status OnnxTestCase::ConvertInput(const std::vector<onnx::TensorProto>& input_pbs, std::unordered_map<std::string, MLValue>& out) {
  int len = static_cast<int>(input_value_info_.size());
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
        std::string vname = input_value_info_[i].name();
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
        if (use_number_names && std::find_if(input_value_info_.begin(), input_value_info_.end(), [buf](const onnx::ValueInfoProto& info) {
                                  return info.name() == buf;
                                }) == input_value_info_.end()) use_number_names = false;
        if (use_data_number_names && std::find_if(input_value_info_.begin(), input_value_info_.end(), [buf2](const onnx::ValueInfoProto& info) {
                                       return info.name() == buf2;
                                     }) == input_value_info_.end()) use_data_number_names = false;
        if (use_data_number_names && std::find_if(input_value_info_.begin(), input_value_info_.end(), [buf3](const onnx::ValueInfoProto& info) {
                                       return info.name() == buf3;
                                     }) == input_value_info_.end()) use_gpu_data_number_names = false;
      }
    }
    for (size_t input_index = 0; input_index != input_pbs.size(); ++input_index) {
      std::string name = var_names[input_index];
      char buf[64];
      if (name.empty()) {
        if (use_number_names) {
          snprintf(buf, sizeof(buf), "%d", static_cast<int>(input_index));
          var_names[input_index] = buf;
        } else if (use_data_number_names) {
          snprintf(buf, sizeof(buf), "data_%d", static_cast<int>(input_index));
          var_names[input_index] = buf;
        } else if (use_gpu_data_number_names) {
          snprintf(buf, sizeof(buf), "gpu_0/data_%d", static_cast<int>(input_index));
          var_names[input_index] = buf;
        } else
          return Status(LOTUS, NOT_IMPLEMENTED, "cannot guess a valid input name");
      }
    }
  }
  for (size_t input_index = 0; input_index != input_pbs.size(); ++input_index) {
    std::string name = var_names[input_index];
    const onnx::TensorProto& input = input_pbs[input_index];
    MLValue v1;
    LOTUS_RETURN_IF_ERROR(Utils::TensorProtoToMLValue(input, allocator_, nullptr, 0, v1));
    out.insert(std::make_pair(name, v1));
  }
  return Status::OK();
}

OnnxTestCase::OnnxTestCase(const AllocatorPtr& allocator, const std::string& test_case_name) : test_case_name_(test_case_name) {
  SetAllocator(allocator);
}

void OnnxTestCase::SetAllocator(const AllocatorPtr& allocator) {
  allocator_ = allocator;
}
