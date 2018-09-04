
#include "core/graph/initializer.h"
#include "core/graph/conv_mul_fusion.h"

using namespace onnx;
using namespace ::onnxruntime::common;
namespace onnxruntime {

bool ConvMulFusion::SatisfyCondition(const Node& node) {
  if (node.OpType() != "Mul" ||
      node.GetInputEdgesCount() != 1 || (*node.InputEdgesBegin())->GetNode().OpType() != "Conv") {
    return false;
  }

  const auto& conv_node = (*node.InputEdgesBegin())->GetNode();
  const auto& conv_inputs = conv_node.InputDefs();
  // For now, fusion is only done when conv has bias.
  if (conv_inputs.size() != 3) {
    return false;
  }
  return true;
}

Status ConvMulFusion::Apply(GraphEditor* graph_editor, Node* node, bool* modified) {
  const auto& conv_node = (*node->InputEdgesBegin())->GetNode();
  const auto& conv_inputs = conv_node.InputDefs();
  auto conv_W_tensor_proto = graph_editor->GetInitializedTensor(conv_inputs[1]->Name());
  auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
  auto conv_B_tensor_proto = graph_editor->GetInitializedTensor(conv_inputs[2]->Name());
  auto conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);

  const auto& mul_inputs = node->InputDefs();
  auto mul_B = std::make_unique<Initializer>(graph_editor->GetInitializedTensor(mul_inputs[1]->Name()));

  ONNXRUNTIME_RETURN_IF_NOT(conv_W->dims().size() > 2 && conv_W->dims()[0] == mul_B->dims()[0]);
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->size() == mul_B->size(), "size is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_W->data_type() == mul_B->data_type(), "data type is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->data_type() == mul_B->data_type(), "data type is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
                            conv_B->data_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
                            "data type is not float or double");
  // Caculate new value of initializers of conv node
  conv_W->scale_by_axis(*mul_B, 1);
  conv_B->mul(*mul_B);

  // Create new initializers of conv
  ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);
  conv_W->ToProto(&new_conv_W_tensor_proto);
  ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto(*conv_B_tensor_proto);
  conv_B->ToProto(&new_conv_B_tensor_proto);

  *modified = true;

  // Replace initializers of conv node
  graph_editor->RemoveInitializedTensor(conv_inputs[1]->Name());
  graph_editor->RemoveInitializedTensor(conv_inputs[2]->Name());
  graph_editor->AddInitializedTensor(new_conv_W_tensor_proto);
  graph_editor->AddInitializedTensor(new_conv_B_tensor_proto);

  // Replace the input of the node following mul node
  const NodeArg* mul_output_def = node->OutputDefs()[0];
  const NodeArg* conv_output_def = conv_node.OutputDefs()[0];
  for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
    graph_editor->ReplaceDef((*it)->Index(), mul_output_def, conv_output_def);
  }

  graph_editor->RemoveNode(node->Index());

  return Status::OK();
}

}  // namespace onnxruntime