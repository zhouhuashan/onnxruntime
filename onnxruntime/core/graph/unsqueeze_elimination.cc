
#include "core/graph/unsqueeze_elimination.h"

using namespace onnx;
using namespace ::onnxruntime::common;

namespace onnxruntime {

bool UnsqueezeElimination::SatisfyCondition(const Node& node) {
  if (node.OpType() != "Unsqueeze" || node.GetInputEdgesCount() != 0) {
    return false;
  }

  const onnxruntime::NodeAttributes& attributes = node.GetAttributes();
  const onnx::AttributeProto* attr = &attributes.find("axes")->second;
  if (attr == nullptr) {
    return false;
  }
  return true;
}
Status UnsqueezeElimination::Apply(GraphEditor* graph_editor, Node* node, bool* modified) {
  // Get attribute of "axes"
  const onnxruntime::NodeAttributes& attributes = node->GetAttributes();
  const onnx::AttributeProto* attr = &attributes.find("axes")->second;
  ONNXRUNTIME_ENFORCE(attr->type() == AttributeProto_AttributeType_INTS, "attribute is not float");
  std::vector<int64_t> axes;
  for (int i = 0; i < attr->ints_size(); i++) {
    axes.push_back(static_cast<int64_t>(attr->ints(i)));
  }

  // Generate new dims
  NodeArg* input_def = node->MutableInputDefs()[0];
  const onnx::TensorProto* tensor_proto = graph_editor->GetInitializedTensor(input_def->Name());
  std::vector<int64_t> new_dims(axes.size() + tensor_proto->dims().size(), 0);

  for (int64_t axis : axes) {
    new_dims[axis] = 1;
  }

  auto begin = tensor_proto->dims().cbegin();
  for (auto& axis : new_dims) {
    if (axis == 0) {
      axis = *begin++;
    }
  }

  // Update shape of tensor proto
  ONNX_NAMESPACE::TensorProto new_tensor_proto(*tensor_proto);
  for (int i = 0; i < new_dims.size(); i++) {
    if (i < tensor_proto->dims().size()) {
      new_tensor_proto.set_dims(i, new_dims[i]);
    } else {
      new_tensor_proto.add_dims(new_dims[i]);
    }
  }
  graph_editor->RemoveInitializedTensor(input_def->Name());
  graph_editor->AddInitializedTensor(new_tensor_proto);

  // Update shape of NodeArg
  TensorShapeProto shape;
  for (auto dim : new_dims) {
    shape.add_dim()->set_dim_value(dim);
  }
  input_def->SetShape(shape);

  // Replace the input of the nodes following unsqueeze node
  const NodeArg* output_def = node->OutputDefs()[0];
  for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
    graph_editor->ReplaceDef((*it)->Index(), output_def, input_def);
  }

  // Remove the Unsqueeze node.
  graph_editor->RemoveNode(node->Index());

  *modified = true;

  return Status::OK();
}
}  // namespace onnxruntime