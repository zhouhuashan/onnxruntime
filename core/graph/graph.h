#ifndef COMMONIR_GRAPH_H
#define COMMONIR_GRAPH_H

#include <string>

#include "core/protobuf/graph.pb.h"

namespace CommonIR
{
    typedef DataProto_DenseTensorProto DenseTensorProto;
    typedef uint32_t NODEINDEX;
#define NODEINDEX_INVALID UINT32_MAX

    typedef int64_t GRAPH_VERSION;
    typedef std::unordered_map<std::string, AttributeProto> NodeAttributes;

    class Graph;

    // Node argument definition, for both input and output,
    // including arg name, arg type and arg shape.
    class NodeArg
    {
    public:

        // Constructor by specifying a name, type and shape.
        NodeArg(const std::string& p_name, const TypeProto& p_type,
            const TensorShapeProto& p_shape);

        // Get node arg name.
        const std::string& Name() const;

        // Get node arg type.
        const TypeProto& Type() const;

        // Get node arg shape.
        const TensorShapeProto& Shape() const;

    private:

        friend class Node;

        // Constructor by specifying a <NodeProto_InputOutputProto>.
        // This is called when loading a <Graph> from <GraphProto> normally.
        NodeArg(const NodeProto_InputOutputProto& p_nodeProtoInputOutput);

        NodeProto_InputOutputProto m_nodeArgData;
    };

    // Function representation.
    // It includes basic function information and its body - a subgraph.
    class Function
    {
    public:

        // Get function body - a subgraph.
        Graph* Body();

        // Get the protobuf representation of <*this> function.
        const FunctionDefProto& Proto()
        {
            return m_functionDefProto;
        }

    private:

        friend class Graph;

        Function() = delete;

        // Constructor.
        explicit Function(const FunctionDefProto& p_funcProto,
            GRAPH_VERSION p_version);

        // Function body which is a SubGraph.
        std::unique_ptr<Graph> m_body;

        // Not owned by <*this> Function, but owned by <Graph>.
        FunctionDefProto m_functionDefProto;
    };

    // A node representation class.
    class Node {

    public:

        // An edge end. It could be input or output edge end of a node.
        // For node's input edge end, it's the source end, as the destination
        // end is the node itself.
        // For node's ouput edge end, it's the destination end, as the source
        // end is the node itself.
        class EdgeEnd
        {
        public:

            // Constructor.
            // An EdgeEnd contains a Node pointer, a NodeArg pointer.
            // NOTE: it does not own the Node pointer and NodeArg pointer.
            EdgeEnd(const Node*& p_node, const NodeArg*& p_nodeArg)
                : m_node(p_node),
                m_nodeArg(p_nodeArg)
            {
            }

            // Get the <Node*> that this edge end refers to. 
            const Node* GetNode() const
            {
                return m_node;
            }

            // Get the <NodeArg*> that this edge end refers to.
            const NodeArg* GetNodeArg() const
            {
                return m_nodeArg;
            }

        private:

            const Node* m_node;

            const NodeArg* m_nodeArg;
        };

        // An iterator helper class for iterating a Node's neighbour nodes.
        class NodeConstIterator
        {
        public:

            NodeConstIterator(std::set<Node*>::const_iterator p_iter)
                : m_iter(p_iter)
            {
            }

            bool operator==(const NodeConstIterator& p_other) const
            {
                return m_iter == p_other.m_iter;
            }

            bool operator!=(const NodeConstIterator& p_other) const
            {
                return m_iter != p_other.m_iter;
            }

            void operator++()
            {
                ++m_iter;
            }

            const Node* operator*()
            {
                return *m_iter;
            }

        private:

            std::set<Node*>::const_iterator m_iter;
        };

        // Get node index.
        NODEINDEX Index() const;

        // Get node name.
        const std::string& Name() const;

        // Get node operator type.
        const std::string& OpType() const;

        // Read/Write <*this> node's input args' definition, including name,
        // type and shape.
        const std::vector<std::vector<NodeArg>>& InputDefs() const;
        std::vector<std::vector<NodeArg>>& Mutable_InputDefs();

        // Read/Write <*this> node's output args' definition, including name,
        // type and shape.
        const std::vector<NodeArg>& OutputDefs() const;
        std::vector<NodeArg>& Mutable_OutputDefs();

        // Functions defined to traverse a Graph as below.
        // Read all input nodes of <*this>.
        Node::NodeConstIterator InputNodes_begin() const;
        Node::NodeConstIterator InputNodes_end() const;
        // Read all output nodes of <*this>.
        Node::NodeConstIterator OutputNodes_begin() const;
        Node::NodeConstIterator OutputNodes_end() const;
        // Given input arg name, get the source end of an input edge.
        // Question: Using name to find EdgeEnd matches our protobuf design
        //           idea that using name to hook nodes.However, it's not that
        //           efficient (using string compare). Shall we use NodeArg* to
        //           do the finding please? Like:
        // bool InputEdgeSrcEnd(const NodeArg& p_inputArg,
        //              EdgeEnd*& p_inputEdgeSrcEnd);
        bool InputEdgeSrcEnd(const std::string& p_inputArgName,
            EdgeEnd*& p_inputEdgeSrcEnd);

        // Add a node attribute with specified attribute name and value.
        template <typename T>
        bool AddAttribute(const std::string& p_attrName, const T& p_value)
        {
            // TODO: add implementation.
        }

        // Clear specified node attribute.
        bool ClearAttribute(const std::string& p_attrName)
        {
            return m_attributes.erase(p_attrName) > 0;
        }

        // Get node attributes.
        const NodeAttributes& GetAttributes() const
        {
            return m_attributes;
        }

        // Indicates which device we'll run this node against in runtime.
        // Executor will decide which device that this node will run against
        // and set it properly.
        // TODO: may change the return value type to be an ENUM.
        const std::string& Device() const;
        void SetDevice(const std::string& p_device);

        // Get the corresponding <NodeProto>.
        void ToProto(NodeProto& p_proto) const;

    private:

        friend class Graph;

        // Node could ONLY be constructed and owned by a <Graph>.
        Node() {}
        Node(NODEINDEX p_index) : m_index(p_index) {}
        Node(const Node& p_other);

        void Init(const NodeProto& p_nodeProto);
        void Init(const std::string& p_name,
            const std::string& p_opType,
            const std::vector<std::vector<NodeArg>>& p_inputArgs,
            const std::vector<NodeArg>& p_outputArgs);

        // Node index.
        NODEINDEX m_index;

        // Node name.
        std::string m_name;

        // Node operator type.
        std::string m_opType;

        // Node inputs' definition.
        std::vector<std::vector<NodeArg>> m_inputDefs;

        // Node outputs' definition.
        std::vector<NodeArg> m_outputDefs;

        // Node inputs' instantiation.
        std::unordered_map<std::string, EdgeEnd> m_inputs;
        // Node input nodes, besides input nodes mentioned in <m_inputs> above,
        // it also contains all control input nodes;
        std::set<Node*> m_inputNodes;
        // Control input nodes' names.
        std::set<std::string> m_controlInputs;
        // Node's output nodes.
        std::set<Node*> m_outputNodes;

        // Device.
        std::string m_device;

        // Map from attribute name to attribute.
        // This allows attribute adding and removing.
        NodeAttributes m_attributes;
    };

    // A graph representation class.
    class Graph
    {
    public:

        // An iterator helper to access graph nodes without copy.
        // The iterator itself does not own any data.
        class NodeConstIterator
        {
        public:

            // Constructor.
            NodeConstIterator(NODEINDEX p_currentNodeIndex, Graph* p_graph)
                : m_graph(p_graph),
                m_currentNodeIndex(p_currentNodeIndex)
            {
            }

            bool operator==(const NodeConstIterator& p_other) const;

            bool operator!=(const NodeConstIterator& p_other) const;

            void operator++();

            const Node* operator*();

        private:

            Graph* m_graph;

            // it's the Node Index in <m_nodes> of the <m_graph>.
            NODEINDEX m_currentNodeIndex;
        };

        // Constructor from scratch.
        Graph(const std::string& p_name, GRAPH_VERSION p_version);

        // Constructor: Given a <GraphProto> loaded from model file, construct
        // a <Graph> object.
        Graph(const GraphProto& p_graphProto);

        // Constructor: Given a set of <NodeProto> defined in a function,
        // construct a <Graph> object.
        // Normally the <p_name> could be the parent node name and the
        // <p_version> could be the parent graph's version.
        // Question: will a node defined in a function refers another function
        // please? I (Ke) am assuming we don't allow such case here for now.
        Graph(const std::string& p_name,
            GRAPH_VERSION p_version,
            const NodeProto* const* p_nodeProtos,
            int size);

        // Resolve inner nodes' dependency and validate whether <*this> Graph
        // is in a good shape. It should also change the <m_graphProto>
        // accordingly to make sure it matches to <*this> Graph.
        // Returns true if resolved successfully, false otherwise.
        bool ResolveDependencyAndValidate();

        // Getter and Setter for <m_version>.
        GRAPH_VERSION Version() const;
        void SetVersion(GRAPH_VERSION p_version);

        // Getter and Setter for <m_name>.
        const std::string& Name() const;
        void SetName(const std::string& p_name);

        // Getter and Setter for graph parameters.
        bool GetParamter(const std::string& p_paramName,
            DenseTensorProto& p_value) const;
        void SetParameter(const std::string& p_paramName,
            const DenseTensorProto& p_value);

        // Add or Remove a function definition.
        bool AddFunctionDef(const FunctionDefProto& p_function);
        void RemoveFunctionDef(const std::string& p_functionName);

        // Get node given specific node index.
        const Node* GetNode(NODEINDEX p_nodeIndex) const;

        // Get node iterator to access all effective nodes in the graph.
        Graph::NodeConstIterator Nodes_begin();
        Graph::NodeConstIterator Nodes_end();

        // Max Node Index.
        NODEINDEX MaxNodeIndex() const;

        // Number of nodes in the <Graph>.
        // This is smaller than MaxNodeIndex(), since there may be nodes
        // removed during optimization.
        int NumberOfNodes();

        // Add, remove node from <*this> graph.
        Node* AddNode(const std::string& p_name,
            const std::string& p_opType,
            const std::vector<std::vector<NodeArg>>& p_inputArgs,
            const std::vector<NodeArg>& p_outputArgs);
        Node* AddNode(const Node& p_other);
        bool RemoveNode(NODEINDEX p_nodeIndex);

        // Add control edge into <*this> graph.
        // The <p_dstNodeIndex> node does not consume any data output by
        // <p_srcNodeIndex>, but it's designed to be executed behind.
        bool AddControlEdge(NODEINDEX p_srcNodeIndex, NODEINDEX p_dstNodeIndex);

        // Try to get function with specified <p_nodeIndex>. Return true if the
        // specified node refers to a function, and <p_function> will be the 
        // function; false otherwise, and <p_function> will be unchanged.
        bool TryGetFunction(NODEINDEX p_nodeIndex,
            /*out*/ Function*& p_function) const;

        // Serialize the <Graph> into <GraphProto>.
        const GraphProto& Proto();

        // Inline all function in <*this> and construct <p_graph>
        // without any functions.
        bool InlineAllFunctions(/*out*/Graph*& p_graph) const;

    private:

        Node* AllocateNode();
        void ReleaseNode(NODEINDEX p_nodeIndex);

        // Add node with specified <p_nodeProto>.
        Node* AddNode(const NodeProto& p_nodeProto);

        // Graph nodes.
        // Element in <m_nodes> may be nullptr due to graph optimization.
        std::vector<std::unique_ptr<Node>> m_nodes;

        // Number of nodes.
        // Normally this is smaller than the size of <m_nodes>, as some
        // elements in <m_nodes> may be removed when doing graph optimization,
        // or some elements may be merged, etc.
        int m_numOfNodes;

        // GraphProto to store name, version, parameters.
        // When serilizing <*this> Graph to a GraphProto, the nodes and
        // functions in <Graph> will also be fed into <m_graphProto> so that
        // it's consistent with <*this> graph.
        GraphProto m_graphProto;

        // Graph function definitions.
        std::unordered_map<std::string,
            std::unique_ptr<Function>> m_functionMap;

        // A flag indicates whether <*this> graph is in a good shape or not.
        bool m_isGraphValid;
    };
}

#endif  // COMMONIR_GRAPH_H