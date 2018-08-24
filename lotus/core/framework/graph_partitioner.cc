#include "core/framework/graph_partitioner.h"

#include "core/graph/indexed_sub_graph.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"

using namespace ::Lotus::Common;
namespace Lotus {
Status GraphPartitioner::Partition(LotusIR::Graph& graph) const {
  if (providers_.Empty()) {
    return Status(LOTUS, INVALID_ARGUMENT, "No provider specified.");
  }

  auto kernel_registries = kernel_registry_mgr_.GetAllKernelRegistries();
  for (auto& provider : providers_) {
    // Partitioning <graph> based on provider preference.
    auto sub_graphs = provider->GetCapability(graph, kernel_registries);
    for (auto& sub_graph : sub_graphs) {
      if (1 == sub_graph->nodes.size()) {
        // The <provider> can run a single node in the <graph>.
        auto node = graph.GetNode(sub_graph->nodes[0]);
        if (node->GetExecutionProviderType().empty()) {
          node->SetExecutionProviderType(provider->Type());
        }
      } else {
        // TODO: This needs to be fused.
        // The <provider> can run a fused <sub_graph> in the <graph>.
      }
    }
  }

  for (auto& node : graph.Nodes()) {
    if (!graph.IsSourceNode(node) && !graph.IsSinkNode(node) && node.GetExecutionProviderType().empty()) {
      return Status(LOTUS, FAIL, "Partitioning failed. No execution provider is capable of running node (" + node.Name() + ").");
    }
  }

  return Status::OK();
}
}  // namespace Lotus
