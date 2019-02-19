// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/node_index_info.h"

#include "core/framework/mlvalue_name_idx_map.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/node_arg.h"

namespace onnxruntime {

NodeIndexInfo::NodeIndexInfo(const GraphViewer& graph_viewer, const MLValueNameIdxMap& mlvalue_idx_map)
    : NodeIndexInfo(graph_viewer.Nodes(), graph_viewer.MaxNodeIndex(), mlvalue_idx_map) {}

NodeIndexInfo::NodeIndexInfo(const GraphNodes& graph_nodes, int max_node_index,
                             const MLValueNameIdxMap& mlvalue_idx_map)
    : max_mlvalue_idx_{mlvalue_idx_map.MaxIdx()} {
  Init(graph_nodes, max_node_index, mlvalue_idx_map);
}

static size_t FindMaxNodeIndex(const std::vector<const Node*>& nodes) {
  NodeIndex max = 0;
  std::for_each(nodes.cbegin(), nodes.cend(), [&max](const Node* node) {
      auto idx = node ? node->Index() : -1; if (idx > max) max = idx; });

  return max;
}

NodeIndexInfo::NodeIndexInfo(const std::vector<const Node*>& nodes, const MLValueNameIdxMap& mlvalue_idx_map)
    : max_mlvalue_idx_{mlvalue_idx_map.MaxIdx()} {
  Init(ValidNodes<const std::vector<const Node*>>(nodes), FindMaxNodeIndex(nodes), mlvalue_idx_map);
}

template <typename TValidNodes>
void NodeIndexInfo::Init(const TValidNodes& nodes, NodeIndex max_node_index, const MLValueNameIdxMap& mlvalue_idx_map) {
  std::size_t total_def_count{};

  bool include_missing_optional_defs = true;

  for (const auto& node : nodes) {
    node.ForEachDef(
        [&](const onnxruntime::NodeArg& /*arg*/, bool /*is_input*/) {
          ++total_def_count;
        },
        include_missing_optional_defs);
  }

  // init all to kInvalidEntry
  node_offsets_.resize(max_node_index + 1, kInvalidEntry);
  node_values_.resize(total_def_count, kInvalidEntry);
  int cur_idx = 0;

  for (auto& node : nodes) {
    node_offsets_[node.Index()] = cur_idx;

    node.ForEachDef(
        [&](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
          auto& name = node_arg.Name();
          if (node_arg.Exists()) {
            int index;
            Status status = mlvalue_idx_map.GetIdx(name, index);
            ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
            node_values_[cur_idx] = index;
          }
          // else it's a missing optional input or output so leave the -1

          ++cur_idx;
        },
        include_missing_optional_defs);
  }
}

}  // namespace onnxruntime
