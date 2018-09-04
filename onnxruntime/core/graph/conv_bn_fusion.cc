
#include "core/graph/initializer.h"
#include "core/graph/conv_bn_fusion.h"

using namespace onnx;
using namespace ::onnxruntime::common;
namespace onnxruntime {

bool ConvBNFusion::SatisfyCondition(const Node& node) {
  if (node.OpType() != "BatchNormalization" ||
      node.GetInputEdgesCount() != 1 || (*node.InputEdgesBegin())->GetNode().OpType() != "Conv") {
    return false;
  }

  const auto& conv_node = (*node.InputEdgesBegin())->GetNode();
  const auto& conv_inputs = conv_node.InputDefs();
  // For now, fusion is only done when conv has bias.
  if (conv_inputs.size() != 3) {
    return false;
  }

  const onnxruntime::NodeAttributes& attributes = node.GetAttributes();
  const onnx::AttributeProto* attr = &(attributes.find("epsilon")->second);
  if (attr == nullptr) {
    return false;
  }

  return true;
}
Status ConvBNFusion::Apply(GraphEditor* graph_editor, Node* node, bool* modified) {
  // Get value of attribute epsilon
  const onnxruntime::NodeAttributes& attributes = node->GetAttributes();
  const onnx::AttributeProto* attr = &(attributes.find("epsilon")->second);
  ONNXRUNTIME_ENFORCE(attr->type() == AttributeProto_AttributeType_FLOAT, "attribute is not float");
  float epsilon = static_cast<float>(attr->f());

  // Get initializers of BatchNormalization
  const auto& bn_inputs = node->InputDefs();
  auto bn_scale = std::make_unique<Initializer>(graph_editor->GetInitializedTensor(bn_inputs[1]->Name()));
  auto bn_B = std::make_unique<Initializer>(graph_editor->GetInitializedTensor(bn_inputs[2]->Name()));
  auto bn_mean = std::make_unique<Initializer>(graph_editor->GetInitializedTensor(bn_inputs[3]->Name()));
  auto bn_var = std::make_unique<Initializer>(graph_editor->GetInitializedTensor(bn_inputs[4]->Name()));

  const auto& conv_node = (*node->InputEdgesBegin())->GetNode();
  const auto& conv_inputs = conv_node.InputDefs();
  auto conv_W_tensor_proto = graph_editor->GetInitializedTensor(conv_inputs[1]->Name());
  auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
  auto conv_B_tensor_proto = graph_editor->GetInitializedTensor(conv_inputs[2]->Name());
  auto conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);

  ONNXRUNTIME_RETURN_IF_NOT(bn_scale->size() == bn_var->size(), "size is not same");
  ONNXRUNTIME_RETURN_IF_NOT(bn_scale->data_type() == bn_var->data_type(), "data type is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_W->dims().size() > 2 && conv_W->dims()[0] == bn_scale->dims()[0]);
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->size() == bn_mean->size(), "size is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->data_type() == bn_mean->data_type(), "data type is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->size() == bn_scale->size(), "size is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->data_type() == bn_scale->data_type(), "data type is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->size() == bn_B->size(), "size is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->data_type() == bn_B->data_type(), "data type is not same");
  ONNXRUNTIME_RETURN_IF_NOT(conv_B->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
                            conv_B->data_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
                            "data type is not float or double");
  // Caculate new value of initializers of conv node
  bn_var->add(epsilon);
  bn_var->sqrt();
  bn_scale->div(*bn_var);
  conv_W->scale_by_axis(*bn_scale, 1);
  conv_B->sub(*bn_mean);
  conv_B->mul(*bn_scale);
  conv_B->add(*bn_B);

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

  // Replace the input of the nodes following batch normalization node
  const NodeArg* bn_output_def = node->OutputDefs()[0];
  const NodeArg* conv_output_def = conv_node.OutputDefs()[0];
  for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
    graph_editor->ReplaceDef((*it)->Index(), bn_output_def, conv_output_def);
  }

  // Remove the Identity node.
  graph_editor->RemoveNode(node->Index());

  return Status::OK();
}

}
