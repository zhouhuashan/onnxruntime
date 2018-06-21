#include "TestCase.h"
#include "core/platform/env.h"
#include "core/framework/tensorprotoutils.h"
#include <fstream>
#include <google/protobuf/util/delimited_message_util.h>
#include <memory>
#include "tml.pb.h"

using namespace std::experimental::filesystem::v1;

using namespace Lotus;

Lotus::Common::Status OnnxTestCase::FromTensorProto(const onnx::TensorProto& input, std::unique_ptr<Lotus::MLValue>& value) {
  std::unique_ptr<Tensor> tensor;
  LOTUS_RETURN_IF_ERROR(Lotus::Utils::GetTensorFromTensorProto(input, &tensor, allocator_));
  value = std::make_unique<Lotus::MLValue>();
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
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

static void SortTensorFileNames(std::vector<path>& input_pb_files) {
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

Lotus::Common::Status OnnxTestCase::SetModelPath(const path& m) {
  model_url = m;
  path test_case_dir = m.parent_path();
  for (directory_iterator test_data_set(test_case_dir), end2; test_data_set != end2; ++test_data_set) {
    if (!is_directory(*test_data_set)) {
      continue;
    }
    test_data_dirs.push_back(test_data_set->path());
  }
  //parse model
  onnx::ModelProto model_pb;
  LOTUS_RETURN_IF_ERROR(loadModelFile(m.string(), model_pb));
  const onnx::GraphProto& graph = model_pb.graph();
  if (graph.node().size() == 1) {
    node_name = graph.node()[0].op_type();
  }
  const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto>& input_value_info = graph.input();
  for (int i = 0; i != input_value_info.size(); ++i) {
    input_value_info_.push_back(input_value_info[i]);
  }
  return Status::OK();
}

Lotus::Common::Status OnnxTestCase::FromTensorProto(const std::vector<onnx::TensorProto>& input, std::vector<Lotus::MLValue>& output_values) {
  for (const onnx::TensorProto& pb : input) {
    std::unique_ptr<Lotus::MLValue> value;
    LOTUS_RETURN_IF_ERROR(FromTensorProto(pb, value));
    output_values.emplace_back(*value.get());
  }
  return Status::OK();
}

//load tensors from disk
static std::vector<onnx::TensorProto> LoadTensors(const std::vector<path>& pb_files) {
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
  SortTensorFileNames(input_pb_files);

  std::vector<onnx::TensorProto> input_pbs = LoadTensors(input_pb_files);
  LOTUS_RETURN_IF_ERROR(ConvertInput(input_pbs, feeds));
  return Status::OK();
}

Lotus::Common::Status OnnxTestCase::FromPbFiles(const std::vector<path>& files, std::vector<Lotus::MLValue>& output_values) {
  std::vector<onnx::TensorProto> input_pbs;
  const path pb(".pb");
  const path tpb(".tpb");
  for (const path& f : files) {
    std::ifstream input(f, std::ios::in | std::ios::binary);
    if (!input) {
      return Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "open file failed");
    }
    if (f.extension() == pb) {
      onnx::TensorProto tensor;
      if (!tensor.ParseFromIstream(&input)) {
        return Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "parse file failed");
      }
      std::unique_ptr<Lotus::MLValue> value;
      LOTUS_RETURN_IF_ERROR(FromTensorProto(tensor, value));
      output_values.emplace_back(*value.get());
    } else if (f.extension() == tpb) {
      Lotus::proto::TraditionalMLData data;
      if (!data.ParseFromIstream(&input)) {
        return Status(Lotus::Common::LOTUS, Lotus::Common::FAIL, "parse file failed");
      }
      if (data.has_vector_map_int64_to_float()) {
        std::unique_ptr<VectorMapInt64ToFloat> vec = std::make_unique<VectorMapInt64ToFloat>();
        for (const Lotus::proto::MapInt64ToFloat& i : data.vector_map_int64_to_float().v()) {
          MapInt64ToFloat new_value;
          for (const auto& j : i.v()) {
            new_value[j.first] = j.second;
          }
          vec->push_back(new_value);
        }
        std::unique_ptr<Lotus::MLValue> value;
        value = std::make_unique<Lotus::MLValue>();
        value->Init(vec.release(),
                    DataTypeImpl::GetType<VectorMapInt64ToFloat>(),
                    DataTypeImpl::GetType<VectorMapInt64ToFloat>()->GetDeleteFunc());
        output_values.emplace_back(*value.get());
      } else if (data.has_vector_map_string_to_float()) {
        std::unique_ptr<VectorMapStringToFloat> vec = std::make_unique<VectorMapStringToFloat>();
        for (const Lotus::proto::MapStringToFloat& i : data.vector_map_string_to_float().v()) {
          MapStringToFloat new_value;
          for (const auto& j : i.v()) {
            new_value[j.first] = j.second;
          }
          vec->push_back(new_value);
        }
        std::unique_ptr<Lotus::MLValue> value;
        value = std::make_unique<Lotus::MLValue>();
        value->Init(vec.release(),
                    DataTypeImpl::GetType<VectorMapStringToFloat>(),
                    DataTypeImpl::GetType<VectorMapStringToFloat>()->GetDeleteFunc());
        output_values.emplace_back(*value.get());
      } else {
        return Status(Lotus::Common::LOTUS, Lotus::Common::NOT_IMPLEMENTED, "unknown file type");
      }
    } else {
      return Status(Lotus::Common::LOTUS, Lotus::Common::NOT_IMPLEMENTED, "unknown file type");
    }
  }
  return Status::OK();
}
Lotus::Common::Status OnnxTestCase::LoadOutputData(size_t id, std::vector<Lotus::MLValue>& output_values) {
  if (id >= test_data_dirs.size())
    return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Attempt to load output data from directory id of ", id, ". Num data dirs :", test_data_dirs.size());

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
    LOTUS_RETURN_IF_ERROR(FromTensorProto(input, v1));
    out.insert(std::make_pair(name, *v1.get()));
  }
  return Status::OK();
}

OnnxTestCase::OnnxTestCase(Lotus::AllocatorPtr allocator, const std::string& test_case_name1) : allocator_(allocator), test_case_name(test_case_name1) {
}
