#pragma once
#include "core/graph/function.h"
#include "core/graph/graph_base.h"
#include "core/graph/model.h"

namespace LotusIR {
class Graph;
class Node;
}  // namespace LotusIR

namespace Lotus {

// Function representation class.
class FunctionImpl : public Function {
 public:
  FunctionImpl(const LotusIR::Graph& graph,
               std::unique_ptr<IndexedSubGraph> customized_func);

  virtual const onnx::OpSchema& OpSchema() const override;

  virtual const LotusIR::GraphBase& Body() const override;

  virtual const IndexedSubGraph& GetIndexedSubGraph() const override;

 private:
  const LotusIR::Graph* parent_graph_;
  std::unique_ptr<IndexedSubGraph> customized_func_body_;
  std::unique_ptr<onnx::OpSchema> op_schema_;
  std::unique_ptr<LotusIR::Model> body_;
};

}  // namespace Lotus
