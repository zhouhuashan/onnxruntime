#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/customregistry.h"
#include "core/framework/execution_providers.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
Status KernelRegistryManager::CreateKernel(const onnxruntime::Node& node,
                                           const IExecutionProvider& execution_provider,
                                           const SessionState& session_state,
                                           /*out*/ std::unique_ptr<OpKernel>& op_kernel) const {
  std::lock_guard<std::mutex> lock(lock_);
  if (kernel_registries_.empty()) {
    return Status(LOTUS, FAIL, "Kernel not found.");
  }

  Status status;
  for (auto& registry : kernel_registries_) {
    status = registry->CreateKernel(node, execution_provider, session_state, op_kernel);
    if (status.IsOK()) {
      return status;
    }
  }

  return status;
}

void KernelRegistryManager::RegisterKernels(const ExecutionProviders& execution_providers,
                                            KernelRegistryPriority priority) {
  for (auto& provider : execution_providers)
    RegisterKernelRegistry(provider->GetKernelRegistry(), priority);
}

void KernelRegistryManager::RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry,
                                                   KernelRegistryPriority priority) {
  std::lock_guard<std::mutex> lock(lock_);
  if (nullptr == kernel_registry) {
    return;
  }

  if (priority == KernelRegistryPriority::HighPriority) {
    kernel_registries_.push_front(kernel_registry);
  } else {
    kernel_registries_.push_back(kernel_registry);
  }
}

Status KernelRegistryManager::SearchKernelRegistry(const onnxruntime::Node& node,
                                                   /*out*/ const KernelCreateInfo** kernel_create_info) const {
  std::lock_guard<std::mutex> lock(lock_);
  if (kernel_registries_.empty()) {
    return Status(LOTUS, FAIL, "Kernel def not found.");
  }

  Status status;
  for (auto& registry : kernel_registries_) {
    status = registry->FindKernel(node, kernel_create_info);
    if (status.IsOK()) {
      return status;
    }
  }

  return status;
}

}  // namespace onnxruntime
