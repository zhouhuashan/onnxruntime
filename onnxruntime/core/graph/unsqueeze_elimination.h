#pragma once

#include "core/graph/rewrite_rule.h"

namespace onnxruntime {
class UnsqueezeElimination : public RewriteRule {
 public:
  UnsqueezeElimination() noexcept : RewriteRule("EliminateUnsqueeze", "Eliminate unsequeeze node") {
  }

 private:
  bool SatisfyCondition(const Node& node) override;

  Status Apply(GraphEditor* graph_editor, Node* node, bool* modified) override;
};
}  // namespace onnxruntime
