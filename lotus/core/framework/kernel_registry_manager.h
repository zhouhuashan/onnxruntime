#pragma once
#include <memory>
#include <vector>
#include <list>
#include "core/common/status.h"
#include "core/graph/graph.h"

namespace Lotus {
class KernelRegistry;
class OpKernel;
struct KernelCreateInfo;
class IExecutionProvider;
class SessionState;
enum class KernelRegistryPriority {
  HighPriority,
  LowPriority
};
class KernelRegistryManager {
 public:
  void RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry, KernelRegistryPriority priority);

  Status CreateKernel(const LotusIR::Node& node,
                      const IExecutionProvider* execution_provider,
                      const SessionState& session_state,
                      /*out*/ std::unique_ptr<OpKernel>* op_kernel) const;

  Status SearchKernelRegistry(const LotusIR::Node& node,
                              /*out*/ const KernelCreateInfo** kernel_create_info) const;

  std::vector<const KernelRegistry*> GetAllKernelRegistries() {
    std::vector<const KernelRegistry*> result;
    for (auto& registry : kernel_registries_) {
      result.push_back(registry.get());
    }
    return result;
  }

 private:
  std::list<std::shared_ptr<KernelRegistry>> kernel_registries_;
};
}  // namespace Lotus
