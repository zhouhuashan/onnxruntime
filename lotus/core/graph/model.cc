#ifdef _MSC_VER
#pragma warning(push)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#endif
#include <google/protobuf/io/coded_stream.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "core/graph/model.h"
#include "core/graph/utils.h"

namespace LotusIR {
Model::Model(const std::string& graph_name, bool isOnnx, const ModelMetaData& model_metadata) {
  model_proto_.set_ir_version(onnx::Version::IR_VERSION);
  model_metadata_ = model_metadata;
  for (auto& metadata : model_metadata_) {
    auto prop = model_proto_.add_metadata_props();
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }
  // Set domain_to_version_ to contain related domains with latest version.
  AddImportOpSets(isOnnx);
  graph_.reset(new Graph(graph_name, domain_to_version_));
}

Model::Model(const ModelProto& model_proto) {
  model_proto_ = model_proto;
  for (auto& prop : model_proto_.metadata_props()) {
    model_metadata_[prop.key()] = prop.value();
  }

  if (0 == model_proto_.opset_import_size()) {
    // Operator sets are not specified in this model.
    // Will use global operator store instead.
    AddImportOpSets(false);
  } else {
    for (auto& op_set : model_proto_.opset_import()) {
      domain_to_version_[op_set.domain()] = static_cast<int>(op_set.version());
    }
  }

  if (model_proto_.has_graph()) {
    graph_.reset(new Graph(model_proto_.graph(), domain_to_version_));
  }
}

Version Model::IrVersion() const {
  if (model_proto_.has_ir_version()) {
    return model_proto_.ir_version();
  }
  return kNoVersion;
}

const std::string& Model::ProducerName() const {
  return model_proto_.producer_name();
}

void Model::SetProducerName(const std::string& producer_name) {
  model_proto_.set_producer_name(producer_name);
}

const std::string& Model::ProducerVersion() const {
  return model_proto_.producer_version();
}

void Model::SetProducerVersion(const std::string& producer_version) {
  model_proto_.set_producer_version(producer_version);
}

const std::string& Model::Domain() const {
  return model_proto_.domain();
}

void Model::SetDomain(const std::string& domain) {
  model_proto_.set_domain(domain);
}

Version Model::ModelVersion() const {
  if (model_proto_.has_model_version()) {
    return model_proto_.model_version();
  }
  return kNoVersion;
}

void Model::SetModelversion(LotusIR::Version version) {
  model_proto_.set_model_version(version);
}

const std::string& Model::DocString() const {
  return model_proto_.doc_string();
}

void Model::SetDocString(const std::string& doc_string) {
  model_proto_.set_doc_string(doc_string);
}

const ModelMetaData& Model::MetaData() const {
  return model_metadata_;
}

Graph* Model::MainGraph() {
  return graph_.get();
}

const Graph* Model::MainGraph() const {
  return graph_.get();
}

const ModelProto& Model::ToProto() {
  *(model_proto_.mutable_graph()) = graph_->ToGraphProto();
  return model_proto_;
}

void Model::AddImportOpSets(bool is_ONNX) {
  auto& domain_to_version_range_map = OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  for (auto& domainToVersionRange : domain_to_version_range_map) {
    if (is_ONNX && domainToVersionRange.first.compare(kOnnxDomain) != 0) {
      // Constructing a pure ONNX model.
      // Only ops in ONNX domain should be used.
      continue;
    }

    domain_to_version_[domainToVersionRange.first] = domainToVersionRange.second.second;
    auto opset_id_proto = model_proto_.add_opset_import();
    opset_id_proto->set_domain(domainToVersionRange.first);
    opset_id_proto->set_version(domainToVersionRange.second.second);
  }
}

#ifdef _WIN32
Status Model::Load(const std::wstring& file_path, std::shared_ptr<Model>* p_model) {
  int fd;
  RETURN_IF_ERROR(FileOpenRd(file_path, &fd));
  auto status = Load(fd, p_model);
  RETURN_IF_ERROR(FileClose(fd));
  return status;
}

Status Model::Save(Model& model, const std::wstring& file_path) {
  int fd;
  RETURN_IF_ERROR(FileOpenWr(file_path, &fd));
  auto status = Save(model, fd);
  RETURN_IF_ERROR(FileClose(fd));
  return status;
}

#endif

Status Model::Load(const std::string& file_path, std::shared_ptr<Model>* p_model) {
  int fd;
  RETURN_IF_ERROR(FileOpenRd(file_path, &fd));
  auto status = Load(fd, p_model);
  RETURN_IF_ERROR(FileClose(fd));
  return status;
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ std::shared_ptr<Model>* p_model) {
  ModelProto model_proto;
  bool result = model_proto.ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  (*p_model).reset(new Model(model_proto));
  RETURN_IF_ERROR((*p_model)->MainGraph()->Resolve());

  return Status::OK();
}

Status Model::Save(Model& model, const std::string& file_path) {
  int fd;
  RETURN_IF_ERROR(FileOpenWr(file_path, &fd));
  auto status = Save(model, fd);
  RETURN_IF_ERROR(FileClose(fd));
  return status;
}

using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::ZeroCopyInputStream;

Status Model::Load(int fd, std::shared_ptr<Model>* p_model) {
  if (fd < 0 || nullptr == p_model) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> less than 0 or <p_model> is nullptr.");
  }

  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);

  ModelProto model_proto;
  bool result = model_proto.ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();

  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  (*p_model).reset(new Model(model_proto));
  RETURN_IF_ERROR((*p_model)->MainGraph()->Resolve());

  return Status::OK();
}

Status Model::Save(Model& model, int p_fd) {
  if (p_fd < 0) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> is less than 0.");
  }

  RETURN_IF_ERROR(model.MainGraph()->Resolve());
  auto& model_proto = model.ToProto();
  bool result = model_proto.SerializeToFileDescriptor(p_fd);
  if (result) {
    return Status::OK();
  } else {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf serialization failed.");
  }
}
}  // namespace LotusIR
