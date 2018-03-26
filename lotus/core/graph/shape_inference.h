#pragma once

#include <functional>

#include "core/common/status.h"
#include "core/graph/opsignature.h"

namespace LotusIR {
class Node;
class NodeArg;

// A context to contain information for shape inference function.
// It includes the operator registry, input arguments definition,
// and mutable output arguments, whose shapes needs to be filled.
class InferenceContext {
 public:
  // TODO: Add input tensors into constructor.
  // TODO: An abstract tensor interface will be needed.
  // In some cases, node evaluation will be needed to get output shapes.
  InferenceContext(Node* p_node, const OpSignature* p_op_schema);

  const Node* GetNode() const;

  const OpSignature* GetOp() const;

  const std::vector<NodeArg*>* GetInputs() const;

  std::vector<NodeArg*>* MutableOutputs();

 private:
  Node* node_;

  const OpSignature* op_signature_;
};

// Shape inference function define.
typedef std::function<Status(InferenceContext&)> ShapeInferenceFunc;
}  // namespace LotusIR
