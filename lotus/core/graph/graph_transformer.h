#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/rewrite_rule.h"

namespace LotusIR {

// A graph transformer interface. A graph transformer transforms a graph in-place.
class GraphTransformer {
 public:
  GraphTransformer(const std::string& name, const std::string& desc)
    : name_(name), desc_(desc) {
  }
  
  virtual ~GraphTransformer() = default;

  // Apply <*this> transformation to a specific graph.
  // Transformation happens in place.
  // The return value of "modified" indicates if the graph was modified or not.
  virtual Status Apply(Graph* graph, bool* modified) const = 0;

private:
  const std::string name_;
  const std::string desc_;
};

// A list of graph transformers.
class GraphTransformerManager {
 public:
  GraphTransformerManager(int32_t steps) : steps_(steps) {
  }
  
  // Register a graph transformer.
  Status Register(std::unique_ptr<GraphTransformer> transformer) {
    transformers_.push_back(std::move(transformer));
  }

  // Apply the list of graph transformers registered on the specified graph
  // up to the given number of steps.
  Status ApplyAll(Graph* graph);

  // A helper to set the basic default list of graph transformers. 
  Status SetDefault();
  
 private:
  GraphTransformerManager() = default;
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphTransformerManager);

  std::vector<std::unique_ptr<GraphTransformer>> transformers_;
  const int32_t steps_;
};

// Rule based graph transformer.
// It provides API to register rewrite rules, and API to apply for
// all applicable rules against one graph.

// Represents a IGraphTransformer determined by a set of rewrite-rules.
// The transformer will apply all the rewrite-rules iteratively as determined by
// the underlying rewriting-strategy.
// TODO: Several rewriting-strategies are possible, with different tradeoffs.
// To begin with, we may use a simple, bottom-up, rewriting strategy.
class RuleBasedGraphTransformer : public GraphTransformer {
 public:
  // Register a rewriting rule.
  // TODO (revisit needed): Using OpSignature* here will ask that OpSignature
  // should be stored globally. Otherwise, there will be multiple addresses/pointers
  // for the same operator or function. To avoid this, we may use OpSignature ID
  // as the key, which should be name_domain_version.
  Status Register(const OpSignature* op, std::unique_ptr<RewriteRule> rule) {
    op_to_rules_[op].push_back(std::move(rule));
    return Status::OK();
  }

  // Apply for all applicable rules against one graph.
  Status Apply(Graph* graph, bool* modified) const override {
    UNUSED_PARAMETER(graph);
    UNUSED_PARAMETER(modified);
    LOTUS_NOT_IMPLEMENTED;
    return Status::OK();
  }

 private:
  typedef std::unordered_map<const OpSignature*, std::vector<std::unique_ptr<RewriteRule>>>
    RewriteRuleSet;

  RewriteRuleSet op_to_rules_;
};

// TODO: Design a loose way to register rewrite rules into RuleBasedGraphTransformer.
// Function representation class.
class Function : public GraphBase {
 public:
  // Get <*this> function's schema.
  const OpSchema& GetSchema() const {
    return schema_;
  }

 private:
  OpSchema schema_;
};

// A function-inlining rewrite-rule. The plan with ONNX is to capture most optimizations
// as function-inlining or function-extraction.
class FunctionInliner : public RewriteRule {
 public:
  FunctionInliner(const std::string& name, const std::string& desc,
                  const Function& function)
    : RewriteRule(name, desc) {
    (function);
  }

  virtual Status Apply(GraphEditor graph_editor,
                       Node* node,
                       bool* modified) override {
    (graph_editor);
    (node);
    (modified);
    LOTUS_NOT_IMPLEMENTED;
    return Status::OK();
  }
};

// A function-extraction rewrite-rule is the dual of function-inlining.
// It identifies occurrences of the body of a function-definition and
// replaces it by a call to the function.
class FunctionExtraction : public RewriteRule {
 public:
  FunctionExtraction(const std::string& name, const std::string& desc,
                     const Function& function)
    : RewriteRule(name, desc) {
    (function);
  }

  virtual Status Apply(GraphEditor graph_editor,
                       Node* node,
                       bool* modified) override {
    (graph_editor);
    (node);
    (modified);
    LOTUS_NOT_IMPLEMENTED;
    return Status::OK();
  }
};
}  // namespace LotusIR
