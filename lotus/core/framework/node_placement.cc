#include "core/framework/node_placement.h"
#include "core/graph/indexed_sub_graph.h"

using namespace Lotus::Common;
namespace Lotus {
Status GraphPartitioner::Partition(LotusIR::Graph* graph) const {
  if (nullptr == graph || providers_.empty()) {
    return Status(LOTUS, INVALID_ARGUMENT, "Graph is nullptr or no provider specified.");
  }

  for (auto& provider : providers_) {
    // Partitioning <graph> based on provider preference.
    auto sub_graphs = std::move(provider->GetCapability(*graph, kernel_registry_mgr_));
    for (auto& sub_graph : sub_graphs) {
      if (1 == sub_graph->nodes.size()) {
        // The <provider> can run a single node in the <graph>.
        auto node = graph->GetNode(sub_graph->nodes[0]);
        if (node->GetExecutionProviderType().empty()) {
          node->SetExecutionProviderType(provider->Type());
        }
      } else {
        // TODO: This needs to be fused.
        // The <provider> can run a fused <sub_graph> in the <graph>.
      }
    }
  }

  for (auto& node : graph->Nodes()) {
    if (!graph->IsSourceNode(node) && !graph->IsSinkNode(node) && node.GetExecutionProviderType().empty()) {
      return Status(LOTUS, FAIL, "Partitioning failed. No execution provider is capable of running node (" + node.Name() + ").");
    }
  }

  return Status::OK();
}
}  // namespace Lotus
