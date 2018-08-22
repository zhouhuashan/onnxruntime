#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"

namespace LotusIR {
class Node;
class Graph;
}  // namespace LotusIR

namespace Lotus {
class KernelDef;
class KernelRegistryManager;
class SessionState;

namespace Logging {
class Logger;
}

namespace Helpers {
const KernelDef* GetKernelDef(const KernelRegistryManager& kernel_registry,
                              const LotusIR::Node& node);

const KernelDef* GetKernelDef(const LotusIR::Graph& graph,
                              const KernelRegistryManager& kernel_registry,
                              const LotusIR::NodeIndex node_id);

/*
AllocatorPtr GetAllocator(const SessionState& session_state,
                          const AllocatorInfo& allocator_info);
                          */
}  // namespace Helpers
}  // namespace Lotus
