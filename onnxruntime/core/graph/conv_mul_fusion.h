#pragma once
#include "core/graph/rewrite_rule.h"

namespace onnxruntime {
class ConvMulFusion : public RewriteRule {
 public:
  ConvMulFusion() noexcept : RewriteRule("ConvBNFusion", "Fusing BN into Conv") {
  }

 private:
  bool SatisfyCondition(const Node& node) override;

  Status Apply(GraphEditor* graph_editor, Node* node, bool* modified) override;
};
}  // namespace onnxruntime