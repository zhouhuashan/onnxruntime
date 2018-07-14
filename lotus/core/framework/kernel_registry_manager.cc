#include "core/framework/kernel_registry_manager.h"
#include "core/framework/customregistry.h"

using namespace onnx;
namespace Lotus {
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
                                                   /*out*/ const KernelCreateInfo** kernel_create_info) const {
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
