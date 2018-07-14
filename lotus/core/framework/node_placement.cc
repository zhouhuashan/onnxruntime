#include "core/framework/node_placement.h"
using namespace Lotus::Common;
namespace Lotus {
Status GraphPlacementPlanner::Apply(LotusIR::Graph* graph, bool* modified) const {
  if (provider_preference_.empty())
    return Status(LOTUS, INVALID_ARGUMENT, "No provider preference found, at least CPU provider should be chosed.");
  if (kernels_registries_.empty())
    return Status(LOTUS, INVALID_ARGUMENT, "No kernel registry found.");

  for (auto& node : graph->Nodes()) {
    if (graph->IsSourceNode(node) || graph->IsSinkNode(node))
      continue;

    if (node.GetExecutionProviderType().empty()) {
      bool assigned = false;
      for (auto& provider_type : provider_preference_) {
        for (auto registry : kernels_registries_) {
          if (registry->CanExecutionProviderCreateKernel(node, provider_type)) {
            node.SetExecutionProviderType(provider_type);
            assigned = true;
            *modified = true;
            break;
          }
        }
        if (assigned)
          break;
      }

      if (!assigned) {
        //for cpu ops with float16, we don't have kernels registered.
        //but we will insert cast to float32. so default to cpu execution provider for backward compatible.
        node.SetExecutionProviderType(LotusIR::kCpuExecutionProvider);
      }
    }
  }

  return Status::OK();
}
}  // namespace Lotus
