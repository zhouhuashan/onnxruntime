#pragma once

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/op_kernel.h"
#include "core/graph/schema_registry.h"

namespace Lotus {

class CustomRegistry : public KernelRegistry, public LotusIR::LotusOpSchemaRegistry {
 public:
  CustomRegistry(bool create_func_kernel);

  ~CustomRegistry() = default;

  /**
    * Register a kernel definition together with kernel factory method to this session.
    * If any conflict happened between registered kernel def and built-in kernel def,
    * registered kernel will have higher priority.
    * Call this before invoking Initialize().
    * @return OK if success.
    */
  Common::Status RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, KernelCreateFn kernel_creator);

 private:
  CustomRegistry() = delete;

  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(CustomRegistry);
};

class KernelRegistryManager {
 public:
  void RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry);

  Status CreateKernel(const LotusIR::Node& node,
                      const IExecutionProvider* execution_provider,
                      const SessionState& session_state,
                      /*out*/ std::unique_ptr<OpKernel>* op_kernel) const;

  Status SearchKernelRegistry(const LotusIR::Node& node,
                              /*out*/ const KernelRegistry::KernelCreateInfo** kernel_create_info) const;

  std::vector<const KernelRegistry*> GetAllKernelRegistries() {
    std::vector<const KernelRegistry*> result;
    for (auto& registry : kernel_registries) {
      result.push_back(registry.get());
    }
    return result;
  }

 private:
  std::list<std::shared_ptr<KernelRegistry>> kernel_registries;
};
}  // namespace Lotus
