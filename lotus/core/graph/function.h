#pragma once

#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"

namespace LotusIR {
class GraphBase;
class Graph;
class Node;
}  // namespace LotusIR

namespace Lotus {

// Function representation class.
class Function {
 public:
  virtual ~Function() {}
  virtual const onnx::OpSchema& OpSchema() const = 0;

  virtual const LotusIR::GraphBase& Body() const = 0;

  virtual const IndexedSubGraph& GetIndexedSubGraph() const = 0;
};

std::unique_ptr<Function> MakeFunction(const LotusIR::Graph& graph,
                                       std::unique_ptr<IndexedSubGraph> customized_func);
}  // namespace Lotus
