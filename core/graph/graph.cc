#include "graph.h"

namespace CommonIR
{
    NodeArg::NodeArg(const NodeProto_InputOutputProto& p_nodeProtoInputOutput)
        : m_nodeArgData(p_nodeProtoInputOutput)
    {
    }

    NodeArg::NodeArg(const std::string& p_name,
        const TypeProto& p_type,
        const TensorShapeProto& p_shape)
    {
        m_nodeArgData.set_name(p_name);
        *(m_nodeArgData.mutable_type()) = p_type;
        *(m_nodeArgData.mutable_shape()) = p_shape;
    }

    const std::string& NodeArg::Name() const
    {
        return m_nodeArgData.name();
    }

    const TypeProto& NodeArg::Type() const
    {
        return m_nodeArgData.type();
    }

    const TensorShapeProto& NodeArg::Shape() const
    {
        return m_nodeArgData.shape();
    }

    Function::Function(const FunctionDefProto& p_funcProto,
        GRAPH_VERSION p_version)
        : m_functionDefProto(p_funcProto),
        m_body(new Graph(p_funcProto.name(),
            p_version,
            p_funcProto.node().data(),
            p_funcProto.node().size()))
    {
    }

    Graph* Function::Body()
    {
        return m_body.get();
    }

    Node::Node(const Node& p_other)
    {
        m_name = p_other.m_name;
        m_opType = p_other.m_opType;
        m_inputDefs = p_other.m_inputDefs;
        m_inputs = p_other.m_inputs;
        m_inputNodes = p_other.m_inputNodes;
        m_outputDefs = p_other.m_outputDefs;
        m_outputNodes = p_other.m_outputNodes;
        m_device = p_other.m_device;
    }

    NODEINDEX Node::Index() const
    {
        return m_index;
    }

    const std::string& Node::Name() const
    {
        return m_name;
    }


    const std::string& Node::OpType() const
    {
        return m_opType;
    }

    const std::vector<std::vector<NodeArg>>& Node::InputDefs() const
    {
        return m_inputDefs;
    }

    std::vector<std::vector<NodeArg>>& Node::Mutable_InputDefs()
    {
        return m_inputDefs;
    }

    Node::NodeConstIterator Node::InputNodes_begin() const
    {
        return NodeConstIterator(m_inputNodes.begin());
    }

    Node::NodeConstIterator Node::InputNodes_end() const
    {
        return NodeConstIterator(m_inputNodes.end());
    }

    Node::NodeConstIterator Node::OutputNodes_begin() const
    {
        return NodeConstIterator(m_outputNodes.begin());
    }

    Node::NodeConstIterator Node::OutputNodes_end() const
    {
        return NodeConstIterator(m_outputNodes.end());
    }

    bool Node::InputEdgeSrcEnd(const std::string& p_inputArgName,
        /*out*/EdgeEnd*& p_inputEdgeSrcEnd)
    {
        auto edgeEndIter = m_inputs.find(p_inputArgName);
        if (m_inputs.end() == edgeEndIter)
        {
            // There's no input edge for the specified input argument.
            return false;
        }

        p_inputEdgeSrcEnd = &(edgeEndIter->second);
        return true;
    }

    const std::vector<NodeArg>& Node::OutputDefs() const
    {
        return m_outputDefs;
    }

    std::vector<NodeArg>& Node::Mutable_OutputDefs()
    {
        return m_outputDefs;
    }

    const std::string& Node::Device() const
    {
        return m_device;
    }

    void Node::SetDevice(const std::string& p_device)
    {
        m_device = p_device;
    }

    void Node::ToProto(NodeProto& p_proto) const
    {
        p_proto.set_name(m_name);
        p_proto.set_op_type(m_opType);

        // Fill control input information.
        p_proto.clear_control_input();

        // Fill attributes.
        p_proto.clear_attr();

        // Fill inputs' defitions.
        p_proto.clear_input();

        // Fill outputs' definition.
        p_proto.clear_output();
    }


    void Node::Init(const NodeProto& p_nodeProto)
    {
        m_name = p_nodeProto.name();
        m_opType = p_nodeProto.op_type();

        for (NodeProto_InputListProto inputList : p_nodeProto.input())
        {
            std::vector<NodeArg> tempInputs;
            for (NodeProto_InputOutputProto input : inputList.input())
            {
                tempInputs.push_back(NodeArg(input));
            }
            m_inputDefs.push_back(tempInputs);
        }

        for (NodeProto_InputOutputProto output : p_nodeProto.output())
        {
            m_outputDefs.push_back(NodeArg(output));
        }

        // Attribute assignment.
    }


    void Node::Init(const std::string& p_name,
        const std::string& p_opType,
        const std::vector<std::vector<NodeArg>>& p_inputArgs,
        const std::vector<NodeArg>& p_outputArgs)
    {
        m_name = p_name;
        m_opType = p_opType;
        m_inputDefs = p_inputArgs;
        m_outputDefs = p_outputArgs;
    }

    bool Graph::NodeConstIterator::operator==(
        const Graph::NodeConstIterator& p_other) const
    {
        return (m_graph == p_other.m_graph
            && m_currentNodeIndex == p_other.m_currentNodeIndex);
    }

    bool Graph::NodeConstIterator::operator!=(
        const Graph::NodeConstIterator& p_other) const
    {
        return !(*this == p_other);
    }

    void Graph::NodeConstIterator::operator++()
    {
        while (true)
        {
            m_currentNodeIndex++;
            if (m_currentNodeIndex < m_graph->MaxNodeIndex()
                && nullptr != m_graph->GetNode(m_currentNodeIndex))
            {
                return;
            }
        }
    }

    const Node* Graph::NodeConstIterator::operator*()
    {
        return m_graph->GetNode(m_currentNodeIndex);
    }

    Graph::Graph(const GraphProto& p_graphProto)
        : m_graphProto(p_graphProto)
    {
        for (auto function : p_graphProto.function())
        {
            m_functionMap[function.name()] =
                std::unique_ptr<Function>(
                    new Function(function, p_graphProto.version()));
        }

        for (auto nodeProto : p_graphProto.node())
        {
            AddNode(nodeProto);
        }

        bool success = ResolveDependencyAndValidate();
        if (false == success)
        {
            // throw exception.
        }
    }

    Graph::Graph(const std::string& p_name,
        GRAPH_VERSION p_version,
        const NodeProto* const* p_nodeProtos,
        int size)
    {
        m_graphProto.set_name(p_name);
        m_graphProto.set_version(p_version);

        for (int i = 0; i < size; ++i)
        {
            AddNode(*(p_nodeProtos[i]));
        }

        bool success = ResolveDependencyAndValidate();
        if (false == success)
        {
            // throw exception.
        }
    }

    Graph::Graph(const std::string& p_name, GRAPH_VERSION p_version)
    {
        m_graphProto.set_name(p_name);
        m_graphProto.set_version(p_version);
    }

    bool Graph::ResolveDependencyAndValidate()
    {
        // TODO: add implementation.
        return true;
    }

    GRAPH_VERSION Graph::Version() const
    {
        return m_graphProto.version();
    }

    void Graph::SetVersion(GRAPH_VERSION p_version)
    {
        m_graphProto.set_version(p_version);
    }

    const std::string& Graph::Name() const
    {
        return m_graphProto.name();
    }

    void Graph::SetName(const std::string& p_name)
    {
        m_graphProto.set_name(p_name);
    }

    bool Graph::GetParamter(const std::string& p_paramName,
        DenseTensorProto& p_value) const
    {
        auto params = m_graphProto.params();

        auto iter = params.find(p_paramName);
        if (params.end() == iter)
        {
            return false;
        }
        p_value = iter->second;
        return true;
    }

    void Graph::SetParameter(const std::string& p_paramName,
        const DenseTensorProto& p_value)
    {
        (*(m_graphProto.mutable_params()))[p_paramName] = p_value;
    }

    bool Graph::AddFunctionDef(const FunctionDefProto& p_function)
    {
        auto funcKey = p_function.name();
        if (m_functionMap.end() != m_functionMap.find(funcKey))
        {
            // Same function exists.
            return false;
        }
        m_functionMap[funcKey] = std::unique_ptr<Function>(
            new Function(p_function, m_graphProto.version()));
        return true;
    }

    void Graph::RemoveFunctionDef(const std::string& p_functionName)
    {
        m_functionMap.erase(p_functionName);
    }

    const Node* Graph::GetNode(NODEINDEX p_nodeIndex) const
    {
        if (MaxNodeIndex() >= p_nodeIndex)
        {
            return nullptr;
        }

        return m_nodes[p_nodeIndex].get();
        return nullptr;
    }

    Graph::NodeConstIterator Graph::Nodes_begin()
    {
        return Graph::NodeConstIterator(0, this);
    }

    Graph::NodeConstIterator Graph::Nodes_end()
    {
        return Graph::NodeConstIterator(MaxNodeIndex(), this);
    }

    NODEINDEX Graph::MaxNodeIndex() const
    {
        return m_nodes.size();
    }

    int Graph::NumberOfNodes()
    {
        return m_numOfNodes;
    }

    Node* Graph::AddNode(const NodeProto& p_nodeProto)
    {
        auto node = AllocateNode();
        node->Init(p_nodeProto);
        return node;
    }

    Node* Graph::AddNode(const std::string& p_name,
        const std::string& p_opType,
        const std::vector<std::vector<NodeArg>>& p_inputArgs,
        const std::vector<NodeArg>& p_outputArgs)
    {
        auto node = AllocateNode();
        node->Init(p_name, p_opType, p_inputArgs, p_outputArgs);
        return node;
    }

    Node* Graph::AddNode(const Node& p_other)
    {
        auto node = AllocateNode();
        *node = p_other;
        return node;
    }

    bool Graph::RemoveNode(NODEINDEX p_index)
    {
        if (MaxNodeIndex() >= p_index)
        {
            return false;
        }

        ReleaseNode(p_index);
        return true;
    }

    bool Graph::AddControlEdge(NODEINDEX p_srcNodeIndex,
        NODEINDEX p_dstNodeIndex)
    {
        if (MaxNodeIndex() >= p_srcNodeIndex
            || MaxNodeIndex() >= p_dstNodeIndex)
        {
            // Invalid node indexes specified.
            return false;
        }
        m_nodes[p_srcNodeIndex]->
            m_outputNodes.insert(m_nodes[p_dstNodeIndex].get());
        m_nodes[p_srcNodeIndex]->
            m_inputNodes.insert(m_nodes[p_srcNodeIndex].get());
    }

    bool Graph::TryGetFunction(NODEINDEX p_index, Function*& p_function) const
    {
        if (MaxNodeIndex() >= p_index)
        {
            return false;
        }

        auto iter = m_functionMap.find(m_nodes[p_index]->OpType());
        if (m_functionMap.end() != iter)
        {
            p_function = iter->second.get();
            return true;
        }
        return false;
    }

    const GraphProto& Graph::Proto()
    {
        // Nodes.
        m_graphProto.clear_node();
        for (auto& node : m_nodes)
        {
            if (nullptr == node)
            {
                continue;
            }
            auto nodeProto = m_graphProto.add_node();
            node->ToProto(*nodeProto);
        }

        // Functions.
        m_graphProto.clear_function();
        for (auto& func : m_functionMap)
        {
            auto funcDef = m_graphProto.add_function();
            (*funcDef) = func.second->Proto();
        }

        return m_graphProto;
    }

    bool Graph::InlineAllFunctions(/*out*/Graph*& p_graph) const
    {
        // TODO: add implementation.
        return true;
    }

    // This pair of functions will be refactored to reuse the node released.
    Node* Graph::AllocateNode()
    {
        std::unique_ptr<Node> node(new Node(MaxNodeIndex()));
        m_nodes.push_back(std::move(node));
        return node.get();
    }

    void Graph::ReleaseNode(NODEINDEX p_nodeIndex)
    {
        m_nodes[p_nodeIndex] = nullptr;
    }
}