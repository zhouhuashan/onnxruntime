#pragma once
#include "core/common/common.h"
#include "core/graph/function.h"
#include "core/graph/rewrite_rule.h"

namespace LotusIR {
class Node;
}  // namespace LotusIR

namespace Lotus {

// A function-inlining rewrite-rule.
class FunctionInliner : public LotusIR::RewriteRule {
 public:
  FunctionInliner(const std::string& name, const std::string& desc)
      : RewriteRule(name, desc) {}

  Status Apply(LotusIR::GraphEditor /*graph_editor*/, LotusIR::Node* /*node*/, bool* /*modified*/) override {
    return Status::OK();
  }
};

}  // namespace Lotus
