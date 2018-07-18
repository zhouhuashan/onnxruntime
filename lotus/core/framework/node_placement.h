#pragma once
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
class GraphPlacementPlanner : public LotusIR::GraphTransformer {
 public:
  //The order of kernel registries represents the priority.
  //The order of provider types represents the user preference.
  GraphPlacementPlanner(const std::string& name,
                        const std::vector<const KernelRegistry*>& registries,
                        const std::vector<std::string>& providers)
      : LotusIR::GraphTransformer(name, "Split the graph according to the kernel registrations."),
        kernels_registries_(registries),
        provider_preference_(providers) {}

  Status Apply(LotusIR::Graph* graph, bool* modified) const override;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphPlacementPlanner);

  std::vector<const KernelRegistry*> kernels_registries_;
  std::vector<std::string> provider_preference_;
};
}  // namespace Lotus
