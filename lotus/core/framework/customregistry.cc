#include "core/framework/customregistry.h"
using namespace onnx;
namespace Lotus {

CustomRegistry::CustomRegistry(bool create_func_kernel) : KernelRegistry(create_func_kernel) {}
Common::Status CustomRegistry::RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, KernelCreateFn kernel_creator) {
  return Register(kernel_def_builder, kernel_creator);
}

void KernelRegistryManager::RegisterKernelRegistry(std::shared_ptr<KernelRegistry> custom_registry) {
  kernel_registries.push_front(custom_registry);
}

Status KernelRegistryManager::CreateKernel(const LotusIR::Node& node,
                                           const IExecutionProvider* execution_provider,
                                           const SessionState& session_state,
                                           /*out*/ std::unique_ptr<OpKernel>* op_kernel) const {
  if (kernel_registries.empty()) {
    return Status(LOTUS, FAIL, "Kernel not found.");
  }

  Status status;
  for (auto& registry : kernel_registries) {
    status = registry->CreateKernel(node, execution_provider, session_state, op_kernel);
    if (status.IsOK()) {
      return status;
    }
  }

  return status;
}

Status KernelRegistryManager::SearchKernelRegistry(const LotusIR::Node& node,
                                                   /*out*/ const KernelRegistry::KernelCreateInfo** kernel_create_info) const {
  if (kernel_registries.empty()) {
    return Status(LOTUS, FAIL, "Kernel def not found.");
  }

  Status status;
  for (auto& registry : kernel_registries) {
    status = registry->SearchKernelRegistry(node, kernel_create_info);
    if (status.IsOK()) {
      return status;
    }
  }

  return status;
}

}  // namespace Lotus
