#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

class ExecutionProviders;
class KernelRegistryManager;

class GraphPartitioner {
 public:
  //The order of providers represents the user preference.
  GraphPartitioner(const KernelRegistryManager& kernel_registry_mgr,
                   const ExecutionProviders& providers)
    : kernel_registry_mgr_(kernel_registry_mgr),
      providers_(providers) {}

  Status Partition(LotusIR::Graph& graph) const;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphPartitioner);

  const KernelRegistryManager& kernel_registry_mgr_;
  const ExecutionProviders& providers_;
};
}  // namespace Lotus
