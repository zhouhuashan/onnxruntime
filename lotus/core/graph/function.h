#pragma once

#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/graph_base.h"

namespace LotusIR {
class Graph;
class Node;
}  // namespace LotusIR

namespace Lotus {

// Function representation class.
class Function {
 public:
  Function(LotusIR::Graph& graph,
           std::unique_ptr<IndexedSubGraph> customized_func,
           std::vector<std::unique_ptr<LotusIR::Node>>& func_nodes);

  const onnx::OpSchema& OpSchema() const;

 private:
  LotusIR::Graph* parent_graph_;
  std::vector<std::unique_ptr<LotusIR::Node>> func_nodes_;
  std::unique_ptr<IndexedSubGraph> customized_func_body_;
  std::unique_ptr<onnx::OpSchema> op_schema_;
};

}  // namespace Lotus
