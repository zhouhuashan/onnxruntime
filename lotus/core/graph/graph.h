#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "core/graph/utils.h"
#include "core/protobuf/onnx-ml.pb.h"

using namespace onnx;

namespace lotusrt {
class LotusRT;
};

namespace Lotus {
namespace Test {
class TestUtils;
}
}  // namespace Lotus

namespace LotusIR {
typedef size_t NodeIndex;
typedef int64_t Version;
typedef ValueInfoProto NodeArgInfo;
typedef std::unordered_map<std::string, TensorProto> InitializedTensorSet;
typedef std::unordered_map<std::string, TypeProto> ArgNameToTypeMap;

class Graph;
class GraphBase;
class Node;
class OpSignature;

// Node argument definition, for both input and output,
// including arg name, arg type (contains both type and shape).
//
// Design Question: in my (Ke's) opinion, shape should not be part of type.
// We may align the protobuf design with our operator registry interface,
// which has type specified for each operator, but no shape. Well, shape
// should be inferred with a separate shape inference function given
// input shapes, or input tensor data sometimes.
// With shape as part of type (current protobuf design),
// 1) we'll have to split the "TypeProto" into type and shape in this internal
// representation interface so that it could be easily used when doing type
// inference and matching with operator registry.
// 2) SetType should be always called before SetShape, otherwise, SetShape()
// will fail. Because shape is located in a TypeProto.
// Thoughts?
//
class NodeArg {
 public:
  // Constructor by specifying node arg name and type&shape which is
  // optional. This is called when loading a <Graph> from <GraphProto>
  // normally.
  NodeArg(const std::string& name,
          const TypeProto* p_arg_type);

  // Get node arg name.
  const std::string& Name() const;

  // Get node arg type.
  const PTYPE Type() const;

  void SetType(PTYPE p_type);
  void SetType(const TypeProto& type_proto);

  // Get node arg shape.
  // Return null pointer if there's no shape specified.
  const TensorShapeProto* Shape() const;

  // Set node arg shape.
  // Shape could only be set after setting type since shape information
  // now is part of TypeProto.
  void SetShape(const TensorShapeProto& shape);

  // Get node arg info proto.
  const NodeArgInfo& ToProto() const;

  // Indicates whether <*this> node arg exists or not.
  // Optional inputs are allowed in ONNX. Empty arg name represents
  // a non-existing input argument.
  bool Exists() const;

 private:
  friend class Node;
  friend class GraphBase;
  friend class Graph;
  friend class lotusrt::LotusRT;

  // Node arg PType.
  PTYPE type_;

  // Node arg name, type and shape.
  NodeArgInfo node_arg_info_;

  // Flag indicates whether <*this> node arg exists or not.
  bool exists_;
};

// A node representation class.
class Node {
 public:
  // An edge end. It could be input or output edge end of a node.
  // For node's input edge end, it's the source end, as the destination
  // end is the node itself.
  // For node's ouput edge end, it's the destination end, as the source
  // end is the node itself.
  class EdgeEnd {
   public:
    // Constructor.
    // An EdgeEnd contains a Node pointer, a NodeArg pointer.
    // NOTE: it does not own the Node pointer and NodeArg pointer.
    // TODO: Can these values ever be null, or should they be references? Would also make the ownership explicit.
    EdgeEnd(const Node* p_node, const NodeArg* p_nodeArg);

    // Get the <Node*> that this edge end refers to.
    const Node* GetNode() const;

    // Get the <NodeArg*> that this edge end refers to.
    const NodeArg* GetNodeArg() const;

   private:
    const Node* node_;

    const NodeArg* node_arg_;
  };

  // An iterator helper class for iterating a Node's neighbour nodes.
  // TODO(Task:134): Determine if Node::NodeConstIterator is really required.
  class NodeConstIterator {
   public:
    NodeConstIterator(std::set<const Node*>::const_iterator iter);

    bool operator==(const NodeConstIterator& other) const;

    bool operator!=(const NodeConstIterator& other) const;

    void operator++();

    const Node* operator*();

   private:
    std::set<const Node*>::const_iterator iter_;
  };

  // Get node index.
  NodeIndex Index() const;

  // Get node name.
  const std::string& Name() const;

  // Get node operator type.
  const std::string& OpType() const;

  // Get the domain of the OperatorSet that specifies the operator named by <op_type_>.
  const std::string& Domain() const;

  // Get the OperatorSchema this node refers to.
  const OperatorSchema* Op() const;

  // Get node description.
  const std::string& Description() const;

  // Read/Write <*this> node's input args' definition, including name,
  // type and shape.
  const std::vector<NodeArg*>& InputDefs() const;
  std::vector<NodeArg*>& MutableInputDefs();

  const std::vector<int>& InputArgCount() const;
  std::vector<int>& MutableInputArgCount();

  // Read/Write <*this> node's output args' definition, including name,
  // type and shape.
  const std::vector<NodeArg*>& OutputDefs() const;
  std::vector<NodeArg*>& MutableOutputDefs();

  // Functions defined to traverse a Graph as below.
  // Read all input nodes of <*this>.
  Node::NodeConstIterator InputNodesBegin() const;
  Node::NodeConstIterator InputNodesEnd() const;
  // Read all output nodes of <*this>.
  Node::NodeConstIterator OutputNodesBegin() const;
  Node::NodeConstIterator OutputNodesEnd() const;

  // Get input/output edges.
  const std::set<EdgeEnd*>& InputEdges() const;
  const std::set<EdgeEnd*>& OutputEdges() const;

  // Given input arg, get the source end of an input edge.
  bool InputEdgeSrcEnd(NodeArg* p_input_arg, /*out*/ const EdgeEnd** pp_input_edge_src_end) const;

  // Add a node attribute with specified attribute name and value.
  bool AddAttribute(const std::string& attr_name, const AttributeProto& value);

#define ADD_ATTR_INTERFACES(TypeName)             \
  bool AddAttribute(const std::string& attr_name, \
                    const TypeName& value);       \
  bool AddAttribute(const std::string& attr_name, \
                    const std::vector<TypeName>& values);

  ADD_ATTR_INTERFACES(int64_t)
  ADD_ATTR_INTERFACES(float)
  ADD_ATTR_INTERFACES(std::string)
  ADD_ATTR_INTERFACES(TensorProto)
  ADD_ATTR_INTERFACES(GraphProto)

  // Clear specified node attribute.
  bool ClearAttribute(const std::string& attr_name);

  // Get node attributes.
  const NodeAttributes& GetAttributes() const;

  // Indicates on which we will run this node in runtime.
  // Executor will decide which device that this node will run against
  // and set it properly.
  // TODO: may change the return value type to be an ENUM.
  const std::string& GetExecutionProvider() const;
  void SetExecutionProvider(const std::string& execution_provider);

  // Get the corresponding <NodeProto>.
  void ToProto(NodeProto& proto) const;

 private:
  friend class GraphBase;
  friend class Graph;

  // Node could ONLY be constructed and owned by a <Graph>.
  Node() {}
  Node(NodeIndex index, GraphBase* p_graph)
      : index_(index),
        graph_(p_graph) {}
  Node(const Node& p_other);

  // Init node per <NodeProto>.
  // <p_nameToValueInfoMap> specifies the node's inputs'/outputs' value information,
  // including name, type and shape.
  void Init(const NodeProto& node_proto,
            const ArgNameToTypeMap& name_to_type);
  void Init(const std::string& name,
            const std::string& op_type,
            const std::string& description,
            const std::vector<NodeArg*>& input_args,
            const std::vector<NodeArg*>& output_args,
            const std::string& domain);

  // Node index.
  NodeIndex index_;

  // Node name.
  std::string name_;

  // Node operator type.
  std::string op_type_;

  // OperatorSet domain of <op_type_).
  std::string domain_;

  // OperatorSchema that <*this> node refers to.
  const OperatorSchema* op_;

  // Node doc string.
  std::string description_;

  // Node inputs' definition.
  std::vector<NodeArg*> input_defs_;
  // The number of inputs for each argument of the operator or function which
  // this node refers.
  // For example, <input_defs_> has 10 elements (inputs), and <input_arg_count_>
  // is {4, 6}. This means that 4 elements (inputs) of <input_defs_> map to the
  // first argument of the operator or function, and the other 6 map to the
  // second argument.
  std::vector<int> input_arg_count_;

  // Node outputs' definition.
  std::vector<NodeArg*> output_defs_;

  // Node inputs edges.
  std::set<EdgeEnd*> input_edges_;
  // Node output edges.
  std::set<EdgeEnd*> output_edges_;
  // Node input nodes, besides input nodes mentioned in <inputs_> above,
  // it also contains all control input nodes;
  std::set<const Node*> input_nodes_;
  // Control input nodes' names.
  std::set<std::string> control_inputs_;
  // Node's output nodes.
  std::set<const Node*> output_nodes_;

  // Device.
  std::string execution_provider_;

  // Map from attribute name to attribute.
  // This allows attribute adding and removing.
  NodeAttributes attributes_;

  GraphBase* graph_;
};

// TODO: Graph base class.
// It should cover the common things between function and graph.
// Move these common things from Graph to GraphBase.
// 1. Graph does not have attributes, while function has.
// 2. Graph does have initializers, while function does not.
// 3. Graph does have value_info, while function does not.
class GraphBase {
 public:
  // An iterator helper to access graph nodes without copy.
  // The iterator itself does not own any data.
  class NodeIterator {
   public:
    // Constructor.
    NodeIterator(NodeIndex current_node_index, GraphBase* p_graph)
        : graph_(p_graph),
          current_node_index_(current_node_index) {
    }

    bool operator==(const NodeIterator& other) const;

    bool operator!=(const NodeIterator& other) const;

    void operator++();

    Node* operator*();

   private:
    GraphBase* graph_;

    // it's the Node Index in <m_nodes> of the <graph_>.
    NodeIndex current_node_index_;
  };

  GraphBase() = default;

  // Resolve <*this> graph to ensure it's in a good shape with full
  // functionality.
  // 1. Run through all validation rules.
  //    a. Node name and node output's names should be unique.
  //    b. Attribute match between node and op definition.
  //    c. Input/Output match between node and op definition.
  //    d. Graph is acyclic and sort nodes in topological order.
  // 2. Check & Setup inner nodes' dependency.
  // 3. Cleanup function definition lists.
  // Returns resolving status.
  virtual Status Resolve() = 0;

  // Getter and Setter for graph name.
  virtual const std::string& Name() const = 0;
  virtual void SetName(const std::string& name) = 0;

  virtual const std::string& Description() const = 0;
  virtual void SetDescription(const std::string& description) = 0;

  // Get graph inputs/outputs.
  virtual const std::vector<const NodeArg*>& GetInputs() const = 0;
  virtual const std::vector<const NodeArg*>& GetOutputs() const = 0;

  virtual std::unordered_map<std::string, NodeArg*>* GetNodeArgMap() = 0;

  // Get node given specific node index.
  Node* GetNode(NodeIndex node_index);
  const Node* GetNode(NodeIndex node_index) const;

  // Get node iterator to access all effective nodes in the graph.
  NodeIterator NodesBegin();
  NodeIterator NodesEnd();

  // Max Node Index.
  NodeIndex MaxNodeIndex() const;

  // Number of nodes in the <Graph>.
  // This is smaller than MaxNodeIndex(), since there may be nodes
  // removed during optimization.
  int NumberOfNodes() const;

  // Add, remove node from <*this> graph.
  Node* AddNode(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const std::string& domain = "");

  /**
  Copy node and add to graph.
  @param other Node to copy
  @param returns Pointer to node that was created and inserted.
  */
  Node* AddNode(const Node& other);

  bool RemoveNode(NodeIndex node_index);

  // Convenience method for adding a constant op
  Node* AddConstantNode(const std::string& name,
                        const std::string& description,
                        const std::vector<NodeArg*>& output_args,
                        const TensorProto& tensor_proto);

  // Add control edge into <*this> graph.
  // The <dst_node_index> node does not consume any data output by
  // <src_node_index>, but it's designed to be executed behind.
  bool AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index);

  bool IsSourceNode(NodeIndex index) const;
  bool IsSinkNode(NodeIndex index) const;

  const Node* SourceNode() const;
  const Node* SinkNode() const;

  // TODO(Task:135) See if GraphBase::GetNodesInTopologicalOrder can be made const
  Status GetNodesInTopologicalOrder(/*out*/ const std::vector<NodeIndex>** pp_nodes);

 protected:
  friend class Node;

  // Add source/sink nodes to <*this> graph.
  void AddSourceSinkNodes();

  // Add node with specified <node_proto>.
  Node* AddNode(const NodeProto& node_proto,
                const ArgNameToTypeMap& name_to_type);

  // Graph nodes.
  // Element in <m_nodes> may be nullptr due to graph optimization.
  std::vector<std::unique_ptr<Node>> nodes_;

  // Number of nodes.
  // Normally this is smaller than the size of <m_nodes>, as some
  // elements in <m_nodes> may be removed when doing graph optimization,
  // or some elements may be merged, etc.
  int num_of_nodes_;

  NodeIndex source_node_index_;
  NodeIndex sink_node_index_;

  // A flag indicates whether <*this> graph needs to be resolved.
  bool graph_resolve_needed_;

  bool graph_proto_sync_needed_;

  int graph_type_ = 0;

  // The topologic order of node index.
  std::vector<NodeIndex> nodes_in_topological_order_;

  // Graph inputs.
  std::vector<const NodeArg*> graph_inputs_;

  // Graph outputs.
  std::vector<const NodeArg*> graph_outputs_;

  const std::unordered_map<std::string, int>* domain_to_version_;

 private:
  // need custom versions to handle the unique_ptr's in nodes_
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphBase);

  Node* AllocateNode();

  void ReleaseNode(NodeIndex node_index);
};

// A graph representation class.
class Graph : public GraphBase {
 public:
  // An iterator helper to access graph nodes without copy.
  // The iterator itself does not own any data.
  //class NodeIterator
  //{
  //public:

  //    // Constructor.
  //    NodeIterator(NodeIndex current_node_index, Graph* p_graph)
  //        : graph_(p_graph),
  //        current_node_index_(current_node_index)
  //    {
  //    }

  //    bool operator==(const NodeIterator& other) const;

  //    bool operator!=(const NodeIterator& other) const;

  //    void operator++();

  //    Node* operator*();

  //private:

  //    Graph* graph_;

  //    // it's the Node Index in <m_nodes> of the <graph_>.
  //    NodeIndex current_node_index_;
  //};

  // Resolve <*this> graph to ensure it's in a good shape with full
  // functionality.
  // 1. Run through all validation rules.
  //    a. Node name and node output's names should be unique.
  //    b. Attribute match between node and op definition.
  //    c. Input/Output match between node and op definition.
  //    d. Graph is acyclic and sort nodes in topological order.
  // 2. Check & Setup inner nodes' dependency.
  // 3. Cleanup function definition lists.
  // Returns resolving status.
  virtual Status Resolve() override;

  // Getter and Setter for graph name.
  virtual const std::string& Name() const override;
  virtual void SetName(const std::string& name) override;

  virtual const std::string& Description() const override;
  virtual void SetDescription(const std::string& description) override;

  virtual std::unordered_map<std::string, NodeArg*>* GetNodeArgMap() override;

  // Add/Remove/Get initial tensors for some graph inputs.
  void AddInitializedTensor(const TensorProto& tensor_proto);
  void RemoveInitializedTensor(const std::string& tensor_name);
  bool GetInitializedTensor(const std::string& tensor_name,
                            TensorProto& value) const;
  const InitializedTensorSet& GetAllInitializedTensors() const;
  void CleanAllInitializedTensors();

  // Get graph inputs/outputs/valueinfos.
  virtual const std::vector<const NodeArg*>& GetInputs() const override;
  virtual const std::vector<const NodeArg*>& GetOutputs() const override;
  const std::vector<const NodeArg*>& GetValueInfo() const;

  // Performs reverse DFS traversal from a set of nodes in 'from' up to
  // the SOURCE node. 'enter' is a visit function that will be invoked
  // on a node when it is visited but its parents haven't been. 'leave'
  // is the visit function invoked on the node after its parents have
  // all been visited. 'comp' is used to stable the traversal order.
  void ReverseDFSFrom(
      const std::vector<NodeIndex>& from,
      const std::function<void(Node*)>& enter,
      const std::function<void(Node*)>& leave,
      const std::function<bool(const Node*, const Node*)>& comp = {});

  void ReverseDFSFrom(
      const std::vector<Node*>& from,
      const std::function<void(Node*)>& enter,
      const std::function<void(Node*)>& leave,
      const std::function<bool(const Node*, const Node*)>& comp = {});

  // Get node given specific node index.
  //Node* GetNode(NodeIndex node_index);

  // Get node iterator to access all effective nodes in the graph.
  //Graph::NodeIterator NodesBegin();
  //Graph::NodeIterator NodesEnd();

  // Max Node Index.
  //NodeIndex MaxNodeIndex() const;

  // Number of nodes in the <Graph>.
  // This is smaller than MaxNodeIndex(), since there may be nodes
  // removed during optimization.
  // int NumberOfNodes() const;

  // Add, remove node from <*this> graph.
  //Node* AddNode(const std::string& name,
  //    const std::string& op_type,
  //    const std::string& description,
  //    const std::vector<NodeArg>& input_args,
  //    const std::vector<NodeArg>& output_args,
  //    const std::string& domain = "");
  //Node* AddNode(const std::string& name,
  //    const std::string& op_type,
  //    const std::string& description,
  //    const std::vector<NodeArg>& input_args,
  //    const std::vector<int>& p_inputArgCount,
  //    const std::vector<NodeArg>& output_args,
  //    const std::string& domain = "");
  //Node* AddNode(const std::string& name,
  //    const std::string& op_type,
  //    const std::string& description,
  //    const std::vector<NodeArg>& output_args,
  //    const std::string& domain = "");
  //Node* AddNode(const Node& other);
  //bool RemoveNode(NodeIndex node_index);

  //// Convenience method for adding a constant op
  //Node* AddConstantNode(const std::string& name,
  //    const std::string& description,
  //    const std::vector<NodeArg>& output_args,
  //    const TensorProto& tensor_proto);

  // Add control edge into <*this> graph.
  // The <dst_node_index> node does not consume any data output by
  // <src_node_index>, but it's designed to be executed behind.
  //bool AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index);

  // Serialize the <Graph> into <GraphProto>.
  const GraphProto& ToGraphProto();

  //bool IsSourceNode(NodeIndex index) const;
  //bool IsSinkNode(NodeIndex index) const;

  //const Node* SourceNode() const;
  //const Node* SinkNode() const;

  //Status GetNodesInTopologicalOrder(std::vector<NodeIndex>** nodes);

 private:
  friend class Model;

  Graph() = delete;

  // Constructor from scratch.
  // going to construct a ONNX graph. With ONNX graph, strict
  // type checking will be skipped.
  Graph(const std::string& name,
        const std::unordered_map<std::string, int>& domain_to_version);

  // Constructor: Given a <GraphProto> loaded from model file, construct
  // a <Graph> object.
  Graph(const GraphProto& graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version);

  enum Type {
    // A main graph.
    Main = 1,
    // A sub graph (function).
    Sub = 2,
  };

  friend class Node;

  Status VerifyNoDuplicateName(
      /*out*/ std::unordered_map<std::string, Node*>& output_args,
      /*out*/ std::unordered_map<std::string, NodeIndex>& node_name_to_index);

  // Build and verify node connection (edges).
  // Verify NodeArg name/type/shape matching correctly.
  Status BuildConnections(
      const std::unordered_map<std::string, Node*>& output_args,
      const std::unordered_map<std::string, NodeIndex>& node_name_to_index);

  // Check whether <*this> graph is acyclic.
  // Depth-first going thru the graph and check whether there's any back
  // edge.
  // <nodes_in_topological_order> returns nodes' indexes in toplogical
  // order if <Status> returned is "OK", otherwise it's undefined.
  Status CheckIsAcyclic(
      /*out*/ std::vector<NodeIndex>& nodes_in_topological_order);

  // Given nodes in topological order, infer and set type information
  // across <*this> graph if needed, and verify type/attribute
  // information match between node and op.
  Status VerifyNodeAndOpMatch(
      const std::vector<NodeIndex>& nodes_in_topological_order,
      std::unordered_map<std::string, Node*>& output_args);

  Status InferAndVerifyTypeMatch(Node* p_node,
                                 const OpSignature* p_op,
                                 const std::unordered_map<std::string, Node*>& output_args);

  // Set graph inputs/outputs when resolving a graph..
  Status SetGraphInputsOutputs();

  // Sync graph inputs/outputs when serializing to proto.
  void SyncGraphInputsOutputs();

  // Graph nodes.
  // Element in <m_nodes> may be nullptr due to graph optimization.
  //std::vector<std::unique_ptr<Node>> m_nodes;

  // Number of nodes.
  // Normally this is smaller than the size of <m_nodes>, as some
  // elements in <m_nodes> may be removed when doing graph optimization,
  // or some elements may be merged, etc.
  //int m_numOfNodes;

  //NodeIndex m_sourceNodeIndex;
  //NodeIndex m_sinkNodeIndex;

  // GraphProto to store name, version, initializer.
  // When serilizing <*this> Graph to a GraphProto, the nodes and
  // functions in <Graph> will also be fed into <graph_proto_> so that
  // it's consistent with <*this> graph.
  GraphProto graph_proto_;

  // The node which refers to <*this> graph (Function).
  //Node* node_;

  InitializedTensorSet name_to_initial_tensor_;

  // A flag indicates whether <*this> graph needs to be resolved.
  //bool graph_resolve_needed_;

  //bool graph_proto_sync_needed_;

  //int graph_type_ = 0;

  // The topologic order of node index.
  //std::vector<NodeIndex> nodes_in_topological_order_;

  // Graph inputs.
  //std::vector<const NodeArg*> graph_inputs_;

  // Graph outputs.
  //std::vector<const NodeArg*> graph_outputs_;

  // Graph value_info.
  std::vector<const NodeArg*> value_info_;

  // Store NodeArg in this graph
  // QUESTION: what does the key represent here?
  std::unordered_map<std::string, NodeArg*> node_args_;

  //const std::unordered_map<std::string, int>* domain_to_version_;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(Graph);
};
}  // namespace LotusIR
