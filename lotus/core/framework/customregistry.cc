#include "core/framework/customregistry.h"

namespace Lotus {

CustomRegistry::CustomRegistry(bool create_func_kernel) : KernelRegistry(create_func_kernel) {}
Common::Status CustomRegistry::RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, KernelCreateFn kernel_creator) {
  return Register(kernel_def_builder, kernel_creator);
}

Common::Status CustomRegistry::RegisterCustomOpSet(std::vector<OpSchema>& schemas, const std::string& domain, int version) {
  //todo: handle domain version
  LOTUS_RETURN_IF_ERROR(AddDomainToVersion(domain, version));
  for (int i = 0; i < schemas.size(); i++)
    LOTUS_RETURN_IF_ERROR(RegisterOpSchema(schemas[i]));
  return Status::OK();
}

void CustomRegistryManager::RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
  custom_registries.push_front(custom_registry);
}

Status CustomRegistryManager::CreateKernel(const LotusIR::Node& node,
                                           const IExecutionProvider* execution_provider,
                                           const SessionState& session_state,
                                           /*out*/ std::unique_ptr<OpKernel>* op_kernel) const {
  if (custom_registries.empty()) {
    return Status(LOTUS, FAIL, "Kernel not found.");
  }

  Status status;
  for (auto& registry : custom_registries) {
    status = registry->CreateKernel(node, execution_provider, session_state, op_kernel);
    if (status.IsOK()) {
      return status;
    }
  }

  return status;
}

Status CustomRegistryManager::SearchKernelRegistry(const LotusIR::Node& node,
                                                   /*out*/ const KernelRegistry::KernelCreateInfo** kernel_create_info) const {
  if (custom_registries.empty()) {
    return Status(LOTUS, FAIL, "Kernel def not found.");
  }

  Status status;
  for (auto& registry : custom_registries) {
    status = registry->SearchKernelRegistry(node, kernel_create_info);
    if (status.IsOK()) {
      return status;
    }
  }

  return status;
}

bool CustomRegistryManager::HasSchema() const {
  for (auto& registry : custom_registries) {
    if (!registry->get_all_schemas().empty()) {
      return true;
    }
  }

  return false;
}

LotusIR::Domain_To_Version_Map CustomRegistryManager::DomainToVersionMap() const {
  LotusIR::Domain_To_Version_Map domain_version_map;

  // Build the map using each of the registries
  for (auto& registry : custom_registries) {
    for (auto& local_domain : registry->DomainVersionMap()) {
      auto iter = domain_version_map.find(local_domain.first);

      // If the map doesn't yet contain this domain, insert it with this registry's value.
      // Otherwise, merge the existing range in the map.
      if (iter == domain_version_map.end()) {
        domain_version_map.insert(local_domain);
      } else {
        // TODO - how should the minimum be treated?  Same issue in Model::AddImportOpSets.
        iter->second.second = std::max(iter->second.second, local_domain.second.second);
      }
    }
  }

  return domain_version_map;
}

const ONNX_NAMESPACE::OpSchema* CustomRegistryManager::Schema(
    const std::string& key,
    const std::string& domain) const {
  for (auto& registry : custom_registries) {
    auto schema = registry->Schema(key, domain);
    if (schema != nullptr) {
      return schema;
    }
  }

  return nullptr;
}

const ONNX_NAMESPACE::OpSchema* CustomRegistryManager::Schema(
    const std::string& key,
    const int maxInclusiveVersion,
    const std::string& domain) const {
  for (auto& registry : custom_registries) {
    auto schema = registry->Schema(key, maxInclusiveVersion, domain);
    if (schema != nullptr) {
      return schema;
    }
  }

  return nullptr;
}

}  // namespace Lotus
