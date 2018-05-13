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

#include "gsl/pointers"
#include "gsl/gsl_util"

namespace LotusIR {
Model::Model(const std::string& graph_name, bool is_onnx_domain_only, const ModelMetaData& model_metadata) {
  model_proto_.reset(new ModelProto);
  model_proto_->set_ir_version(onnx::Version::IR_VERSION);
  model_proto_->mutable_graph()->set_name(graph_name);
  model_metadata_ = model_metadata;
  for (auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop = model_proto_->add_metadata_props();
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  // Set domain_to_version_ to contain related domains with latest version.
  std::unordered_map<std::string, int> domain_to_version;
  AddImportOpSets(is_onnx_domain_only, &domain_to_version);

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(model_proto_->mutable_graph(), domain_to_version, IrVersion()));
}

Model::Model(const ModelProto& model_proto)
    : Model(std::unique_ptr<ModelProto>(new ModelProto(model_proto))) {
}

Model::Model(std::unique_ptr<ModelProto> model_proto) {
  assert(nullptr != model_proto);
  model_proto_.reset(model_proto.release());
  for (auto& prop : model_proto_->metadata_props()) {
    model_metadata_[prop.key()] = prop.value();
  }

  std::unordered_map<std::string, int> domain_to_version;
  if (0 == model_proto_->opset_import_size()) {
    // Operator sets are not specified in this model.
    // Will use global operator store instead.
    AddImportOpSets(false, &domain_to_version);
  } else {
    for (auto& opSet : model_proto_->opset_import()) {
      domain_to_version[opSet.domain()] = static_cast<int>(opSet.version());
    }
  }

  if (model_proto_->has_graph()) {
    graph_.reset(new Graph(model_proto_->mutable_graph(), domain_to_version, IrVersion()));
  }
}

Version Model::IrVersion() const {
  if (model_proto_->has_ir_version()) {
    return model_proto_->ir_version();
  }
  return kNoVersion;
}

const std::string& Model::ProducerName() const {
  return model_proto_->producer_name();
}

void Model::SetProducerName(const std::string& producer_name) {
  model_proto_->set_producer_name(producer_name);
}

const std::string& Model::ProducerVersion() const {
  return model_proto_->producer_version();
}

void Model::SetProducerVersion(const std::string& producer_version) {
  model_proto_->set_producer_version(producer_version);
}

const std::string& Model::Domain() const {
  return model_proto_->domain();
}

void Model::SetDomain(const std::string& domain) {
  model_proto_->set_domain(domain);
}

Version Model::ModelVersion() const {
  if (model_proto_->has_model_version()) {
    return model_proto_->model_version();
  }
  return kNoVersion;
}

void Model::SetModelversion(LotusIR::Version version) {
  model_proto_->set_model_version(version);
}

const std::string& Model::DocString() const {
  return model_proto_->doc_string();
}

void Model::SetDocString(const std::string& doc_string) {
  model_proto_->set_doc_string(doc_string);
}

const ModelMetaData& Model::MetaData() const noexcept {
  return model_metadata_;
}

Graph* Model::MainGraph() noexcept {
  return graph_.get();
}

const Graph* Model::MainGraph() const noexcept {
  return graph_.get();
}

ModelProto Model::ToProto() {
  *(model_proto_->mutable_graph()) = graph_->ToGraphProto();
  return *model_proto_;
}

void Model::AddImportOpSets(bool is_onnx_domain_only, /*out*/ std::unordered_map<std::string, int>* domain_to_version) {
  auto& domain_to_version_range_map = OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  for (auto& domainToVersionRange : domain_to_version_range_map) {
    if (is_onnx_domain_only && domainToVersionRange.first.compare(kOnnxDomain) != 0) {
      // Constructing an onnx-domain-only model.
      // Only ops in ONNX domain should be used.
      continue;
    }

    domain_to_version->insert({domainToVersionRange.first, domainToVersionRange.second.second});
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto = model_proto_->add_opset_import();
    opset_id_proto->set_domain(domainToVersionRange.first);
    opset_id_proto->set_version(domainToVersionRange.second.second);
  }
}

Status Model::Load(const ModelProto& model_proto, gsl::not_null<std::shared_ptr<Model>*> model) {
  // we expect a graph to be present
  if (!model_proto.has_graph()) {
    return Status(LOTUS, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  model->reset(new Model(model_proto));

  if ((*model)->MainGraph() != nullptr) {
    RETURN_IF_ERROR((*model)->MainGraph()->Resolve(true));
  }
  return Status::OK();
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

Status Model::Load(const std::string& file_path, gsl::not_null<std::shared_ptr<Model>*> p_model) {
  int fd;
  RETURN_IF_ERROR(FileOpenRd(file_path, &fd));
  auto status = Load(fd, p_model);
  RETURN_IF_ERROR(FileClose(fd));
  return status;
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ gsl::not_null<std::shared_ptr<Model>*> p_model) {
  std::unique_ptr<ModelProto> modelProto(new ModelProto);
  bool result = modelProto->ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  (*p_model).reset(new Model(std::move(modelProto)));
  if ((*p_model)->MainGraph() != nullptr) {
    RETURN_IF_ERROR((*p_model)->MainGraph()->Resolve(true));
  }
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

Status Model::Load(int fd, gsl::not_null<std::shared_ptr<Model>*> p_model) {
  if (fd < 0) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> less than 0.");
  }

  auto raw_input = std::unique_ptr<ZeroCopyInputStream>(std::make_unique<FileInputStream>(fd));
  auto coded_input = std::make_unique<CodedInputStream>(raw_input.get());

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);

  std::unique_ptr<ModelProto> model_proto(new ModelProto);
  const bool result = model_proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();

  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  (*p_model).reset(new Model(std::move(model_proto)));
  if ((*p_model)->MainGraph() != nullptr) {
    RETURN_IF_ERROR((*p_model)->MainGraph()->Resolve(true));
  }
  return Status::OK();
}

Status Model::Save(Model& model, int p_fd) {
  if (p_fd < 0) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> is less than 0.");
  }

  RETURN_IF_ERROR(model.MainGraph()->Resolve());
  auto model_proto = model.ToProto();
  const bool result = model_proto.SerializeToFileDescriptor(p_fd);
  if (result) {
    return Status::OK();
  } else {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf serialization failed.");
  }
}
}  // namespace LotusIR
