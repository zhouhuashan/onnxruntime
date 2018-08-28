#include "core/graph/function.h"
#include "core/graph/graph.h"
namespace Lotus {
Function::Function(LotusIR::Graph& graph,
                   std::unique_ptr<IndexedSubGraph> customized_func,
                   std::vector<std::unique_ptr<LotusIR::Node>>& func_nodes) {
  for (auto& func_node : func_nodes) {
    func_nodes_.push_back(std::move(func_node));
  }
  parent_graph_ = &graph;
  customized_func_body_ = std::move(customized_func);
  auto meta_def = customized_func_body_->GetMetaDef();
  op_schema_ = std::make_unique<onnx::OpSchema>();
  op_schema_->SetName(meta_def->name);
  op_schema_->SetDomain(meta_def->domain);
  op_schema_->SetDoc(meta_def->doc_string);
  op_schema_->SinceVersion(meta_def->since_version);
  int i = 0;
  for (auto& input : meta_def->inputs) {
    auto input_type = parent_graph_->GetNodeArg(input)->Type();
    op_schema_->Input(i, input, "", *input_type);
    ++i;
  }
  i = 0;
  for (auto& output : meta_def->outputs) {
    auto output_type = parent_graph_->GetNodeArg(output)->Type();
    op_schema_->Output(i, output, "", *output_type);
    ++i;
  }
  op_schema_->Finalize();
}

const onnx::OpSchema& Function::OpSchema() const {
  return *op_schema_;
}

}  // namespace Lotus
