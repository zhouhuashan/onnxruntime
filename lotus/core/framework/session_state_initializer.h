#pragma once
#include <map>

#include "core/framework/allocator.h"
#include "core/framework/tensor.h"

namespace LotusIR {
class Graph;
class GraphTransformerManager;
}  // namespace LotusIR

namespace Lotus {
class SessionState;
class ExecutionProviders;
class KernelRegistryManager;
class InsertCastTransformer;

namespace Logging {
class Logger;
}

class SessionStateInitializer {
 public:
  SessionStateInitializer(LotusIR::Graph& graph,
                          SessionState& session_state,
                          const ExecutionProviders& providers,
                          KernelRegistryManager& kernel_registry_manager,
                          const Logging::Logger& logger);

  // First perform any transformations and create the execution plan
  Common::Status CreatePlan(const LotusIR::GraphTransformerManager& graph_transformation_manager,
                            const InsertCastTransformer& insert_cast_transformer,
                            bool enable_sequential_execution);

  // initialize tensors, and save. save kernels and input/output node mappings
  // @param enable_memory_pattern
  Common::Status InitializeAndSave(bool enable_memory_pattern,
                                   std::map<AllocatorInfo, BufferUniquePtr>& weights_buffers);

 private:
  LotusIR::Graph& graph_;
  SessionState& session_state_;

  const ExecutionProviders& execution_providers_;
  KernelRegistryManager& kernel_registry_manager_;
  const Logging::Logger& logger_;
};
}  // namespace Lotus
