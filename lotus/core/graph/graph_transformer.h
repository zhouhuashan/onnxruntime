#ifndef CORE_GRAPH_GRAPH_TRANSFORMER_H
#define CORE_GRAPH_GRAPH_TRANSFORMER_H

#include "core/graph/graph.h"

namespace LotusIR {
class GraphEditor {
 public:
  explicit GraphEditor(Graph& p_graph) {
    m_graph = &p_graph;
  }
  GraphEditor() = delete;
  GraphEditor(const GraphEditor& p_other) = delete;

  // Add node from <m_graph>.
  Node* AddNode(const std::string& p_name,
                const std::string& p_opType,
                const std::string& p_description,
                const std::vector<NodeArg*>& p_inputArgs,
                const std::vector<NodeArg*>& p_outputArgs,
                const std::string& p_domain = "") {
    return m_graph->AddNode(p_name, p_opType, p_description,
                            p_inputArgs, p_outputArgs, p_domain);
  }

  Node* AddNode(const Node& p_other) {
    return m_graph->AddNode(p_other);
  }

  // Remove node from <m_graph>.
  bool RemoveNode(NODEINDEX p_nodeIndex) {
    return m_graph->RemoveNode(p_nodeIndex);
  }

  // Add control edge into <m_graph>.
  // The <p_dstNodeIndex> node does not consume any data output by
  // <p_srcNodeIndex>, but it's designed to be executed behind.
  bool AddControlEdge(NODEINDEX p_srcNodeIndex, NODEINDEX p_dstNodeIndex) {
    return m_graph->AddControlEdge(p_srcNodeIndex, p_dstNodeIndex);
  }

  // Resolve <m_graph> after each editing.
  Status Resolve() {
    return m_graph->Resolve();
  }

 private:
  Graph* m_graph;
};

// A rewrite-rule interface. A rewrite-rule represents a semantics-preserving transformation of a
// computation-graph. It can be used to represent, for example, the elimination of operators that
// serve as no-ops (for example, dropout during inference), as well as inlining of "function"
// definitions or the dual (replacing a complex expression by an equivalent function-call).
// Unlike the more general IGraphTransformer, a rewrite-rule is applied at a single node,
// representing the root of an expression that is rewritten.
class IRewriteRule {
 public:
  virtual ~IRewriteRule() {}

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
  virtual Status Apply(/*IN/OUT*/ Node& p_node,
                       GraphEditor p_graphEditor,
                       /*OUT*/ bool& modified) = 0;
};

// A graph transformer interface. A graph transformer transforms a graph in-place.
class IGraphTransformer {
 public:
  virtual ~IGraphTransformer() {}

  // Apply <*this> transformation to a specific graph.
  // Transformation happens in place.
  // The return value of "modified" indicates if the graph was modified or not.
  virtual Status Apply(/*IN/OUT*/ Graph& p_graph, /*OUT*/ bool& modified) = 0;
};

class GraphTransformerManager {
 public:
  // Register a graph transformer.
  Status Register(const IGraphTransformer& p_graphTransformer);

  // Going thru all transformers registered in <*this> manager on
  // specified graph.
  Status ApplyAll(/*IN/OUT*/ Graph& p_graph);

  static GraphTransformerManager Instance() {
    static GraphTransformerManager s_graphTransformerMgr;
    return s_graphTransformerMgr;
  }

 private:
  GraphTransformerManager() = default;
  std::vector<IGraphTransformer*> m_transformers;
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
  Status Register(IRewriteRule& p_rule, const std::vector<OpSignature*>& p_ops);

  // Apply for all applicable rules against one graph.
  virtual Status Apply(/*IN/OUT*/ Graph& p_graph, /*OUT*/ bool& modified);

  static RuleBasedGraphTransformer Instance() {
    static RuleBasedGraphTransformer s_ruleBasedGraphTransformer;
    return s_ruleBasedGraphTransformer;
  }

 private:
  RuleBasedGraphTransformer() = default;

  std::unordered_map<OpSignature*, std::vector<IRewriteRule*>> m_opToRules;
};

// TODO: Design a loose way to register rewrite rules into RuleBasedGraphTransformer.
// Function representation class.
class Function : public GraphBase {
 public:
  // Get <*this> function's schema.
  const OperatorSchema& GetSchema() const;

 private:
  OperatorSchema m_schema;
};

// A function-inlining rewrite-rule. The plan with ONNX is to capture most optimizations
// as function-inlining or function-extraction.
class FunctionInliner : public IRewriteRule {
 public:
  FunctionInliner(const Function& function) {
    (function);
  }

  virtual Status Apply(/*IN/OUT*/ Node& p_node,
                       GraphEditor p_graphEditor,
                       /*OUT*/ bool& modified) override {
    (p_node);
    (p_graphEditor);
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

  virtual Status Apply(/*IN/OUT*/ Node& p_node,
                       GraphEditor p_graphEditor,
                       /*OUT*/ bool& modified) override {
    (p_node);
    (p_graphEditor);
    (modified);
    return Status::OK();
  }
};
}  // namespace LotusIR
#endif  // CORE_GRAPH_GRAPH_TRANSFORMER_H
