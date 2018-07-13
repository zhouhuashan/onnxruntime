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

#include <memory>
#include "core/graph/model.h"
#include "core/graph/utils.h"
#include "core/graph/schema_registry.h"
#include "gsl/pointers"
#include "gsl/gsl_util"
using namespace onnx;
using namespace Lotus;
using namespace Lotus::Common;

namespace LotusIR {
Model::Model(const std::string& graph_name, bool is_onnx_domain_only, const ModelMetaData& model_metadata, const ILotusOpSchemaRegistryList* local_registries) {
  model_proto_ = std::make_unique<ModelProto>();
  model_proto_->set_ir_version(onnx::Version::IR_VERSION);
  model_proto_->mutable_graph()->set_name(graph_name);
  model_metadata_ = model_metadata;
  for (auto& metadata : model_metadata_) {
    const gsl::not_null<StringStringEntryProto*> prop = model_proto_->add_metadata_props();
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  auto schema_registry = std::shared_ptr<SchemaRegistryManager>(new SchemaRegistryManager());
  if (local_registries != nullptr) {
    for (auto schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }

  // Set domain_to_version_ to contain related domains with latest version.
  std::unordered_map<std::string, int> domain_to_version = schema_registry->GetLatestOpsetVersions(is_onnx_domain_only);
  for (auto domain : domain_to_version) {
    const gsl::not_null<OperatorSetIdProto*> opset_id_proto = model_proto_->add_opset_import();
    opset_id_proto->set_domain(domain.first);
    opset_id_proto->set_version(domain.second);
  }
  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(model_proto_->mutable_graph(), domain_to_version, IrVersion(), schema_registry));
}

Model::Model(const ModelProto& model_proto, const ILotusOpSchemaRegistryList* local_registries)
    : Model(std::make_unique<ModelProto>(model_proto), local_registries) {
}

Model::Model(std::unique_ptr<ModelProto> model_proto, const ILotusOpSchemaRegistryList* local_registries) {
  assert(nullptr != model_proto);
  model_proto_.reset(model_proto.release());
  for (auto& prop : model_proto_->metadata_props()) {
    model_metadata_[prop.key()] = prop.value();
  }

  auto schema_registry = std::shared_ptr<SchemaRegistryManager>(new SchemaRegistryManager());
  if (local_registries != nullptr) {
    for (auto schema_collection : *local_registries) {
      schema_registry->RegisterRegistry(schema_collection);
    }
  }

  std::unordered_map<std::string, int> domain_to_version;
  if (0 == model_proto_->opset_import_size()) {
    // Operator sets are not specified in this model.
    // Will use global operator store instead.
    domain_to_version = schema_registry->GetLatestOpsetVersions(false);
    for (auto domain : domain_to_version) {
      const gsl::not_null<OperatorSetIdProto*> opset_id_proto = model_proto_->add_opset_import();
      opset_id_proto->set_domain(domain.first);
      opset_id_proto->set_version(domain.second);
    }
  } else {
    for (auto& opSet : model_proto_->opset_import()) {
      domain_to_version[opSet.domain()] = gsl::narrow_cast<int>(opSet.version());
    }
  }

  if (model_proto_->has_graph()) {
    // create instance. need to call private ctor so can't use make_unique
    GSL_SUPPRESS(r .11)
    graph_.reset(new Graph(model_proto_->mutable_graph(), domain_to_version, IrVersion(), schema_registry));
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

Status Model::Load(std::istream& model_istream, ModelProto* p_model_proto) {
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

Status Model::Load(const ModelProto& model_proto, std::shared_ptr<Model>& model, const ILotusOpSchemaRegistryList* local_registries) {
  // we expect a graph to be present
  if (!model_proto.has_graph()) {
    return Status(LOTUS, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  model.reset(new Model(model_proto, local_registries));

  if (model->MainGraph() != nullptr) {
    LOTUS_RETURN_IF_ERROR(model->MainGraph()->Resolve(true));
  }
  return Status::OK();
}

#ifdef _WIN32
GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const std::wstring& file_path, std::shared_ptr<Model>& p_model, const ILotusOpSchemaRegistryList* local_registries) {
  int fd;
  LOTUS_RETURN_IF_ERROR(FileOpenRd(file_path, &fd));
  auto status = Load(fd, p_model, local_registries);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

Status Model::Save(Model& model, const std::wstring& file_path) {
  int fd;
  LOTUS_RETURN_IF_ERROR(FileOpenWr(file_path, &fd));
  auto status = Save(model, fd);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

#endif

GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const std::string& file_path, std::shared_ptr<Model>& p_model, const ILotusOpSchemaRegistryList* local_registries) {
  int fd;
  if (!FileOpenRd(file_path, &fd).IsOK()) {
    return Status(LOTUS, NO_MODEL, "Failed to open: " + file_path);
  }
  auto status = Load(fd, p_model, local_registries);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ std::shared_ptr<Model>& p_model, const ILotusOpSchemaRegistryList* local_registries) {
  std::unique_ptr<ModelProto> modelProto = std::make_unique<ModelProto>();
  const bool result = modelProto->ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  p_model = std::make_shared<Model>(std::move(modelProto), local_registries);
  if (p_model->MainGraph() != nullptr) {
    LOTUS_RETURN_IF_ERROR(p_model->MainGraph()->Resolve(true));
  }
  return Status::OK();
}

Status Model::Save(Model& model, const std::string& file_path) {
  int fd;
  LOTUS_RETURN_IF_ERROR(FileOpenWr(file_path, &fd));
  auto status = Save(model, fd);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::ZeroCopyInputStream;

Status Model::Load(int fd, std::shared_ptr<Model>& p_model, const ILotusOpSchemaRegistryList* local_registries) {
  if (fd < 0) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> less than 0.");
  }

  auto raw_input = std::unique_ptr<ZeroCopyInputStream>(std::make_unique<FileInputStream>(fd));
  auto coded_input = std::make_unique<CodedInputStream>(raw_input.get());

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);

  std::unique_ptr<ModelProto> model_proto = std::make_unique<ModelProto>();
  const bool result = model_proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();

  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  p_model = std::make_shared<Model>(std::move(model_proto), local_registries);
  if (p_model->MainGraph() != nullptr) {
    LOTUS_RETURN_IF_ERROR(p_model->MainGraph()->Resolve(true));
  }
  return Status::OK();
}

Status Model::Save(Model& model, int p_fd) {
  if (p_fd < 0) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> is less than 0.");
  }

  LOTUS_RETURN_IF_ERROR(model.MainGraph()->Resolve());
  auto model_proto = model.ToProto();
  const bool result = model_proto.SerializeToFileDescriptor(p_fd);
  if (result) {
    return Status::OK();
  } else {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf serialization failed.");
  }
}
}  // namespace LotusIR
