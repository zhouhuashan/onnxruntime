#include "TestCase.h"
#include "core/platform/env.h"
#include "core/framework/tensorprotoutils.h"
#include <fstream>
#include <google/protobuf/util/delimited_message_util.h>
#include <memory>
#include "tml.pb.h"

using namespace std::experimental::filesystem::v1;

using namespace Lotus;

template <typename InputType, typename OutputType>
Common::Status ConvertVector(const InputType& data, OutputType** vec) {
  //void* p = allocator->Alloc(sizeof(OutputType));
  //if (p == nullptr)
  //	return Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "out of memory");
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
  return Common::Status::OK();
}

template <typename InputType, typename OutputType>
Common::Status Convert(const InputType& tensor_proto, OutputType** p_tensor, AllocatorPtr allocator);

template <>
Common::Status Convert(const onnx::TensorProto& tensor_proto, Tensor** out, AllocatorPtr allocator) {
  std::unique_ptr<Tensor> p_tensor;
  auto st = Lotus::Utils::GetTensorFromTensorProto(tensor_proto, &p_tensor, allocator);
  if (st.IsOK()) *out = p_tensor.release();
  return st;
}

template <>
Common::Status Convert(const google::protobuf::RepeatedPtrField< ::Lotus::proto::MapInt64ToFloat>& data, VectorMapInt64ToFloat** vec, AllocatorPtr allocator) {
  return ConvertVector<google::protobuf::RepeatedPtrField< ::Lotus::proto::MapInt64ToFloat>, VectorMapInt64ToFloat>(data, vec);
}

template <>
Common::Status Convert(const google::protobuf::RepeatedPtrField< ::Lotus::proto::MapStringToFloat>& data, VectorMapStringToFloat** vec, AllocatorPtr allocator) {
  return ConvertVector<google::protobuf::RepeatedPtrField< ::Lotus::proto::MapStringToFloat>, VectorMapStringToFloat>(data, vec);
}

template <typename InputType, typename OutputType>
Lotus::Common::Status ProtoToMLValue(const InputType& input, std::unique_ptr<Lotus::MLValue>& value, AllocatorPtr allocator) {
  OutputType* tensor = nullptr;
  Common::Status st = Convert(input, &tensor, allocator);
  if (!st.IsOK()) return st;
  value = std::make_unique<Lotus::MLValue>();
  value->Init(tensor,
              DataTypeImpl::GetType<OutputType>(),
              DataTypeImpl::GetType<OutputType>()->GetDeleteFunc());
  return Lotus::Common::Status::OK();
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

static Common::Status SortTensorFileNames(std::vector<path>& input_pb_files) {
  if (input_pb_files.size() <= 1) return Common::Status::OK();
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
  return Common::Status::OK();
}

Lotus::Common::Status loadModelFile(const std::string& model_url, onnx::ModelProto& model_pb) {
  std::ifstream input(model_url, std::ios::in | std::ios::binary);
  if (!input) {
    std::ostringstream oss;
    oss << "open file " << model_url << " failed";
    return Status(Lotus::Common::LOTUS, Lotus::Common::NO_SUCHFILE, oss.str());
  }
  if (!model_pb.ParseFromIstream(&input)) {
    std::ostringstream oss;
    oss << "parse file " << model_url << " failed";
    return Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_PROTOBUF, oss.str());
  }
  return Status::OK();
}

template <typename T>
static void RepeatedPtrFieldToVector(const ::google::protobuf::RepeatedPtrField<T>& input_value_info, std::vector<T>& out) {
  for (int i = 0; i != input_value_info.size(); ++i) {
    out.push_back(input_value_info[i]);
  }
}

Lotus::Common::Status OnnxTestCase::ParseModel() {
  Lotus::Common::Status st = Common::Status::OK();
  std::call_once(model_parsed, [this, &st]() {
    //parse model
    onnx::ModelProto model_pb;
    st = loadModelFile(model_url.string(), model_pb);
    if (!st.IsOK()) return;
    const onnx::GraphProto& graph = model_pb.graph();
    if (graph.node().size() == 1) {
      node_name = graph.node()[0].op_type();
    }
    RepeatedPtrFieldToVector(graph.input(), input_value_info_);
    RepeatedPtrFieldToVector(graph.output(), output_value_info_);
    st = Common::Status::OK();
  });
  return st;
}
Lotus::Common::Status OnnxTestCase::SetModelPath(const path& m) {
  model_url = m;
  path test_case_dir = m.parent_path();
  for (directory_iterator test_data_set(test_case_dir), end2; test_data_set != end2; ++test_data_set) {
    if (!is_directory(*test_data_set)) {
      continue;
    }
    test_data_dirs.push_back(test_data_set->path());
  }
  return Status::OK();
}

//load tensors from disk
static Common::Status LoadTensors(const std::vector<path>& pb_files, std::vector<onnx::TensorProto>* input_pbs) {
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
  return Common::Status::OK();
}

Lotus::Common::Status OnnxTestCase::LoadInputData(size_t id, std::unordered_map<std::string, Lotus::MLValue>& feeds) {
  if (id >= test_data_dirs.size())
    return Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, "out of bound");

  path inputs_pb = test_data_dirs[id] / "inputs.pb";
  if (std::experimental::filesystem::exists(inputs_pb)) {  //has an all-in-one input file
    std::string content;
    LOTUS_RETURN_IF_ERROR(Env::Default().ReadFileAsString(inputs_pb.c_str(), &content));
    std::vector<onnx::TensorProto> input_pbs;
    google::protobuf::io::CodedInputStream coded_input((const uint8_t*)content.data(), (int)content.size());
    bool clean_eof = false;
    onnx::TensorProto tensor;
    while (google::protobuf::util::ParseDelimitedFromCodedStream(&tensor, &coded_input, &clean_eof)) {
      input_pbs.push_back(tensor);
      tensor.Clear();
    }

    if (!clean_eof) {
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "parse input file '", inputs_pb.string(), "' failed, clean_eof==false");
    }
    LOTUS_RETURN_IF_ERROR(ConvertInput(input_pbs, feeds));
    return Status::OK();
  }

  std::vector<path> input_pb_files;
  const path pb(".pb");

  for (directory_iterator pb_file(test_data_dirs[id]), end3; pb_file != end3; ++pb_file) {
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

class TensorDataLoder : public DataLoder {
 private:
  Lotus::AllocatorPtr allocator_;

 public:
  TensorDataLoder(const Lotus::AllocatorPtr& allocator) : allocator_(allocator) {
  }

  Lotus::Common::Status Load(const path& f, std::unique_ptr<Lotus::MLValue>& value) const override {
    onnx::TensorProto tensor;
    {
      std::ifstream input(f, std::ios::in | std::ios::binary);
      if (!input) {
        return Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "open file failed");
      }
      if (!tensor.ParseFromIstream(&input)) {
        return Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "parse file failed");
      }
    }
    auto status = ProtoToMLValue<onnx::TensorProto, Lotus::Tensor>(tensor, value, allocator_);
    if (!status.IsOK()) return status;
    return Status::OK();
  }
};

class TraditionalMLDataLoder : public DataLoder {
 private:
  Lotus::AllocatorPtr allocator_;

 public:
  TraditionalMLDataLoder(const Lotus::AllocatorPtr& allocator) : allocator_(allocator) {
  }
  Common::Status Load(const path& f, std::unique_ptr<Lotus::MLValue>& value) const override {
    Lotus::proto::TraditionalMLData data;
    {
      std::ifstream input(f, std::ios::in | std::ios::binary);
      if (!input) {
        return Common::Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "open file failed");
      }
      if (!data.ParseFromIstream(&input)) {
        return Common::Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "parse file failed");
      }
    }
    Common::Status st;
    switch (data.values_case()) {
      case Lotus::proto::TraditionalMLData::kVectorMapStringToFloat:
        st = ProtoToMLValue<decltype(data.vector_map_string_to_float().v()), VectorMapStringToFloat>(data.vector_map_string_to_float().v(), value, allocator_);
        break;
      case Lotus::proto::TraditionalMLData::kVectorMapInt64ToFloat:
        st = ProtoToMLValue<decltype(data.vector_map_int64_to_float().v()), VectorMapInt64ToFloat>(data.vector_map_int64_to_float().v(), value, allocator_);
        break;
      default:
        st = Status(Lotus::Common::LOTUS, Lotus::Common::NOT_IMPLEMENTED, "unknown data type inside TraditionalMLData");
    }
    return st;
  }
};

Lotus::Common::Status OnnxTestCase::FromPbFiles(const std::vector<path>& files, std::vector<Lotus::MLValue>& output_values) {
  for (const path& f : files) {
    if (!f.has_extension()) return LOTUS_MAKE_STATUS(Lotus::Common::LOTUS, Lotus::Common::NOT_IMPLEMENTED, "unknown file type, path = ", f);
    std::string s = f.extension().string();
    if (s.empty() || s[0] != '.')
      return LOTUS_MAKE_STATUS(Lotus::Common::LOTUS, Lotus::Common::NOT_IMPLEMENTED, "file has no extension, path = ", f);
    auto iter = loaders.find(s.substr(1));
    if (iter == loaders.end()) {
      return LOTUS_MAKE_STATUS(Lotus::Common::LOTUS, Lotus::Common::NOT_IMPLEMENTED, "unknown file extension, path = ", f);
    }
    std::unique_ptr<Lotus::MLValue> v;
    LOTUS_RETURN_IF_ERROR(iter->second->Load(f, v));
    output_values.emplace_back(*v.get());
  }
  return Status::OK();
}

Lotus::Common::Status OnnxTestCase::LoadOutputData(size_t id, std::vector<Lotus::MLValue>& output_values) {
  if (id >= test_data_dirs.size())
    return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, test_case_name, ":Attempt to load output data from directory id of ", id, ". Num data dirs :", test_data_dirs.size());

  std::vector<path> output_pb_files;
  const path pb(".pb");
  const path tpb(".tpb");
  for (directory_iterator pb_file(test_data_dirs[id]), end3; pb_file != end3; ++pb_file) {
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

Lotus::Common::Status OnnxTestCase::ConvertInput(const std::vector<onnx::TensorProto>& input_pbs, std::unordered_map<std::string, Lotus::MLValue>& out) {
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
          return Status(Lotus::Common::LOTUS, Lotus::Common::NOT_IMPLEMENTED, "cannot guess a valid input name");
      }
    }
  }
  for (size_t input_index = 0; input_index != input_pbs.size(); ++input_index) {
    std::string name = var_names[input_index];
    const onnx::TensorProto& input = input_pbs[input_index];
    std::unique_ptr<Lotus::MLValue> v1;
    auto status = ProtoToMLValue<onnx::TensorProto, Lotus::Tensor>(input, v1, allocator_);
    out.insert(std::make_pair(name, *v1.get()));
  }
  return Status::OK();
}

OnnxTestCase::OnnxTestCase(const Lotus::AllocatorPtr& allocator, const std::string& test_case_name1) : allocator_(allocator), test_case_name(test_case_name1) {
  loaders["pb"] = new TensorDataLoder(allocator);
  loaders["tpb"] = new TraditionalMLDataLoder(allocator);
}
