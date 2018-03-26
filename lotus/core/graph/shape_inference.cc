#include "core/graph/shape_inference.h"
#include "core/graph/graph.h"

namespace LotusIR {
InferenceContext::InferenceContext(Node* p_node,
                                   const OpSignature* p_op_schema)
    : node_(p_node),
      op_signature_(p_op_schema) {
}

const Node* InferenceContext::GetNode() const {
  return node_;
}

const OpSignature* InferenceContext::GetOp() const {
  return op_signature_;
}

const std::vector<NodeArg*>* InferenceContext::GetInputs() const {
  if (nullptr == node_) {
    return nullptr;
  }
  return &(node_->InputDefs());
}

std::vector<NodeArg*>* InferenceContext::MutableOutputs() {
  if (nullptr == node_) {
    return nullptr;
  }
  return &(node_->MutableOutputDefs());
}
}  // namespace LotusIR
