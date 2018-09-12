#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
class Node;
class Graph;
}  // namespace onnxruntime

namespace onnxruntime {
class ExecutionProviders;
class KernelDef;
class KernelRegistryManager;
class SessionState;

namespace Logging {
class Logger;
}

namespace Utils {
const KernelDef* GetKernelDef(const KernelRegistryManager& kernel_registry,
                              const onnxruntime::Node& node);

const KernelDef* GetKernelDef(const onnxruntime::Graph& graph,
                              const KernelRegistryManager& kernel_registry,
                              const onnxruntime::NodeIndex node_id);

AllocatorPtr GetAllocator(const ExecutionProviders& exec_providers, const AllocatorInfo& allocator_info);

AllocatorPtr GetAllocator(const SessionState& session_state,
                          const AllocatorInfo& allocator_info);
}  // namespace Utils
}  // namespace onnxruntime
