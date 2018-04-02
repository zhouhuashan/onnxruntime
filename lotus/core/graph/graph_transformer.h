#pragma once

#include "core/graph/graph.h"

#include "core/common/common.h"

namespace LotusIR {
class GraphEditor {
 public:
  explicit GraphEditor(Graph& graph) : graph_{&graph} {}

  // Add node from <graph_>.
  Node* AddNode(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const std::string& domain = "") {
    return graph_->AddNode(name, op_type, description,
                           input_args, output_args, domain);
  }

  Node* AddNode(const Node& other) {
    return graph_->AddNode(other);
  }

  // Remove node from <graph_>.
  bool RemoveNode(NodeIndex node_index) {
    return graph_->RemoveNode(node_index);
  }

  // Add control edge into <graph_>.
  // The <dst> node does not consume any data output by
  // <src>, but it's designed to be executed behind.
  bool AddControlEdge(NodeIndex src, NodeIndex dst) {
    return graph_->AddControlEdge(src, dst);
  }

  // Resolve <graph_> after each editing.
  Status Resolve() {
    return graph_->Resolve();
  }

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphEditor);

  Graph* graph_;
};

// A rewrite-rule interface. A rewrite-rule represents a semantics-preserving transformation of a
// computation-graph. It can be used to represent, for example, the elimination of operators that
// serve as no-ops (for example, dropout during inference), as well as inlining of "function"
// definitions or the dual (replacing a complex expression by an equivalent function-call).
// Unlike the more general IGraphTransformer, a rewrite-rule is applied at a single node,
// representing the root of an expression that is rewritten.
class IRewriteRule {
 public:
  virtual ~IRewriteRule() = default;

  // Rewrite rule name.
  virtual const std::string& Name() const = 0;

  // Rewrite rule description.
  virtual const std::string& Description() const {
    static const std::string description("");
    return description;
  }

  // Apply the rewrite rule to a specific node.
  // The transformation happens in-place. The return-value of node may be different
  // from the input-value due to rewriting.
  // The return value of "modified" indicates if the graph was modified or not.
  virtual Status Apply(/*IN/OUT*/ Node& node,
                       GraphEditor& graph_editor,
                       /*OUT*/ bool& modified) = 0;
};

// A graph transformer interface. A graph transformer transforms a graph in-place.
class IGraphTransformer {
 public:
  virtual ~IGraphTransformer() {}

  // Apply <*this> transformation to a specific graph.
  // Transformation happens in place.
  // The return value of "modified" indicates if the graph was modified or not.
  virtual Status Apply(/*IN/OUT*/ Graph& graph, /*OUT*/ bool& modified) = 0;
};

class GraphTransformerManager {
 public:
  // Register a graph transformer.
  Status Register(const IGraphTransformer& graph_transformer);

  // Going thru all transformers registered in <*this> manager on
  // specified graph.
  Status ApplyAll(/*IN/OUT*/ Graph& graph);

  static GraphTransformerManager& Instance() {
    static GraphTransformerManager s_graph_transformer_manager;
    return s_graph_transformer_manager;
  }

 private:
  GraphTransformerManager() = default;
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphTransformerManager);

  std::vector<IGraphTransformer*> transformers_;
};

// Rule based graph transformer.
// It provides API to register rewrite rules, and API to apply for
// all applicable rules against one graph.

// Represents a IGraphTransformer determined by a set of rewrite-rules.
// The transformer will apply all the rewrite-rules iteratively as determined by
// the underlying rewriting-strategy.
// TODO: Several rewriting-strategies are possible, with different tradeoffs.
// To begin with, we may use a simple, bottom-up, rewriting strategy.
class RuleBasedGraphTransformer : public IGraphTransformer {
 public:
  // Register a rewriting rule.
  // TODO (revisit needed): Using OpSignature* here will ask that OpSignature should be storeed globally,
  // otherwise, there will be multiple adresses/pointers for the same operator or function.
  // To avoid this ask, we may use OpSignature ID as the key, which should be name_domain_version.
  Status Register(IRewriteRule& rule, const std::vector<OpSignature*>& ops);

  // Apply for all applicable rules against one graph.
  virtual Status Apply(/*IN/OUT*/ Graph& graph, /*OUT*/ bool& modified);

  static RuleBasedGraphTransformer Instance() {
    static RuleBasedGraphTransformer s_rule_based_graph_transformer;
    return s_rule_based_graph_transformer;
  }

 private:
  RuleBasedGraphTransformer() = default;

  std::unordered_map<OpSignature*, std::vector<IRewriteRule*>> op_to_rules_;
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
class FunctionInliner : public IRewriteRule {
 public:
  FunctionInliner(const Function& function) {
    (function);
  }

  virtual Status Apply(/*IN/OUT*/ Node& node,
                       GraphEditor& graph_editor,
                       /*OUT*/ bool& modified) override {
    (node);
    (graph_editor);
    (modified);
    return Status::OK();
  }
};

// A function-extraction rewrite-rule is the dual of function-inlining. It identifies
// occurrences of the body of a function-definition and replaces it by a call to the function.
class FunctionExtraction : public IRewriteRule {
 public:
  FunctionExtraction(const Function& function) {
    (function);
  }

  virtual Status Apply(/*IN/OUT*/ Node& node,
                       GraphEditor& graph_editor,
                       /*OUT*/ bool& modified) override {
    (node);
    (graph_editor);
    (modified);
    return Status::OK();
  }
};
}  // namespace LotusIR
