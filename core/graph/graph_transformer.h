#ifndef CORE_GRAPH_GRAPH_TRANSFORMER_H
#define CORE_GRAPH_GRAPH_TRANSFORMER_H

#include "core/graph/graph.h"

namespace LotusIR
{
    class GraphEditor {
    public:

        explicit GraphEditor(Graph& p_graph)
        {
            m_graph = &p_graph;
        }
        GraphEditor() = delete;
        GraphEditor(const GraphEditor& p_other) = delete;

        // Add node from <m_graph>.
        Node* AddNode(const std::string& p_name,
            const std::string& p_opType,
            const std::string& p_description,
            const std::vector<NodeArg>& p_inputArgs,
            const std::vector<NodeArg>& p_outputArgs,
            const std::string& p_domain = "")
        {
            return m_graph->AddNode(p_name, p_opType, p_description, p_inputArgs, p_outputArgs, p_domain);
        }
        Node* AddNode(const Node& p_other)
        {
            return m_graph->AddNode(p_other);
        }

        // Remove node from <m_graph>.
        bool RemoveNode(NODEINDEX p_nodeIndex)
        {
            return m_graph->RemoveNode(p_nodeIndex);
        }

        // Add control edge into <m_graph>.
        // The <p_dstNodeIndex> node does not consume any data output by
        // <p_srcNodeIndex>, but it's designed to be executed behind.
        bool AddControlEdge(NODEINDEX p_srcNodeIndex, NODEINDEX p_dstNodeIndex)
        {
            return m_graph->AddControlEdge(p_srcNodeIndex, p_dstNodeIndex);
        }

        // Resolve <m_graph> after each editing.
        Status Resolve()
        {
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
            return "";
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
    class IGraphTransformer
    {
    public:

        virtual ~IGraphTransformer() {}

        // Apply <*this> transformation to a specific graph.
        // Transformation happens in place.
        // The return value of "modified" indicates if the graph was modified or not.
        virtual Status Apply(/*IN/OUT*/ Graph& p_graph, /*OUT*/ bool& modified) = 0;
    };


    // Rule based graph transformer.
    // It provides API to register rewrite rules, and API to apply for
    // all applicable rules against one graph.

    // Represents a IGraphTransformer determined by a set of rewrite-rules.
    // The transformer will apply all the rewrite-rules iteratively as determined by
    // the underlying rewriting-strategy.
    // TODO: Several rewriting-strategies are possible, with different tradeoffs.
    // To begin with, we may use a simple, bottom-up, rewriting strategy.
    class RuleBasedGraphTransformer : public IGraphTransformer
    {
    public:

        // Register a rewriting rule.
        // TODO (revisit needed): Using OpSignature* here will ask that OpSignature should be storeed globally,
        // otherwise, there will be multiple adresses/pointers for the same operator or function.
        // To avoid this ask, we may use OpSignature ID as the key, which should be name_domain_version.
        Status Register(IRewriteRule& p_rule, const std::vector<OpSignature*>& p_ops);

        // Apply for all applicable rules against one graph.
        virtual Status Apply(/*IN/OUT*/ Graph& p_graph, /*OUT*/ bool& modified);

        static RuleBasedGraphTransformer Instance()
        {
            static RuleBasedGraphTransformer s_ruleBasedGraphTransformer;
            return s_ruleBasedGraphTransformer;
        }

    private:

        RuleBasedGraphTransformer() = default;

        std::unordered_map<OpSignature*, std::vector<IRewriteRule>> m_opToRules;
    };

    //TODO: Design a loose way to register rewrite rules into RuleBasedGraphTransformer.

    // Function representation class.
    class Function : public GraphBase
    {
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
            // TODO
        }

        virtual Status Apply(/*IN/OUT*/ Node& p_node,
            GraphEditor p_graphEditor,
            /*OUT*/ bool& modified) override {
            // TODO
        }
    };

    // A function-extraction rewrite-rule is the dual of function-inlining. It identifies
    // occurrences of the body of a function-definition and replaces it by a call to the function.
    class FunctionExtraction : public IRewriteRule {
    public:
        FunctionExtraction(const Function& function) {
            // TODO
        }

        virtual Status Apply(/*IN/OUT*/ Node& p_node,
            GraphEditor p_graphEditor,
            /*OUT*/ bool& modified) override {
            // TODO
        }
    };

    // TODO: Tensor class design.
    class Tensor;

    // A work space managing all data (inputs/outputs) during graph evaluation.
    class WorkSpace
    {
    public:

        // Get a tensor with specific tensor name for specific execution provider.
        // If copy needed, <*this> work space will call specific execution provider's allocator to do it.
        const Tensor* GetTensor(const std::string& p_tensorName, const std::string& p_executionProviderID) const;

        // Create a tensor with specific tensor name for specific execution provider.
        // It should check that there's no name duplication firstly and then call specific execution 
        // provider's allocator to do the creation.
        Tensor* CreateTensor(const std::string& p_tensorName, const std::string& p_executionProviderID);

    };


    // Work space view for one execution provider.
    // It provides APIs to read (inputs) and write (outputs).
    // It acts as an execution provider view of work space, which do the real memory allocation and copy among different
    // execution providers by call their tensor allocators and porters.
    class WorkSpaceView
    {
    public:

        explicit WorkSpaceView(WorkSpace& p_workSpace, const std::string& p_executionProviderID);

        const Tensor* GetTensor(const std::string& p_tensorName) const;

        Tensor* CreateTensor(const std::string& p_tensorName);

    private:

        WorkSpace* m_workSpace;

        const std::string* m_executionProviderID;
    };

    // Allocator interface. It provides APIs to allocating data (inputs/outputs) and copying data (IN/OUT) for
    // one exeuction provider.
    class IAllocator
    {
    public:

        IAllocator(const std::string& p_exeuctionProviderID)
        {
            m_executionProviderID = &p_exeuctionProviderID;
        }

        enum PortingDirection
        {
            IN = 0,
            OUT = 1,
        };

        virtual Status AllocateTensor(const std::string& p_tensorName, /*OUT*/ Tensor** p_tensor) = 0;

        Status CopyTensor(const Tensor* p_srcTensor, Tensor* p_dstTensor, PortingDirection p_direction)
        {
            if (nullptr == p_srcTensor || nullptr != p_dstTensor)
            {
                return Status(LOTUS, FAIL, "Invalid inputs.");
            }

            switch (p_direction)
            {
            case IN:
                return CopyCPUTensorTo(p_srcTensor, p_dstTensor);
            case OUT:
                return CopyTensorToCPU(p_srcTensor, p_dstTensor);
            default:
                return Status(LOTUS, FAIL, "Invalid inputs.");
                break;
            }
        }

    protected:

        virtual Status CopyCPUTensorTo(const Tensor* p_srcTensor, Tensor* p_dstTensor) = 0;

        virtual Status CopyTensorToCPU(const Tensor* p_srcTensor, Tensor* p_dstTensor) = 0;

        const std::string* m_executionProviderID;
    };

    class IExecutionProvider
    {
    public:

        IExecutionProvider()
        {
            m_id = Name() + "." + Domain() + "." + Version();
        }

        virtual ~IExecutionProvider() {}

        virtual const std::string& Name() const = 0;

        virtual const std::string& Domain() const = 0;

        virtual const std::string& Version() const = 0;

        virtual const std::string& ID() const
        {
            return m_id;
        }

        // Get IAllocator for <*this> execution provider.
        // It will be used for allocating tensors (inputs/outputs) or copying tensors (IN/OUT)
        // for this exeuction provider.
        virtual IAllocator& Allocator() const = 0;

        // TODO: Function should have same concept of OpSignature as core operators.
        // TODO: Create copy constructor from FunctionDefProto to OpSignature.
        virtual bool Support(const OpSignature& p_op, const NodeAttributes& p_attributes) = 0;

        // Execute the <p_node> given <p_workSpace> which contains inputs/outputs for <*this> execution provider.
        virtual Status Execute(const Node& p_node, WorkSpaceView& p_workSpace) = 0;




        // There is a tradeoff between the following two API methods for declaring an execution-provider's capabilities.
        // The first is more general (e.g., if an execution-provider supports an operation with specific attribute-values).
        // The second could be a more efficient approximation that is useful in some contexts: for example, for a
        // graph-transformer that recognizes all occurrences of sub-graphs that can be replaced by a function call.

        // Indicates whether the execution provider can realize/execute a given node.
        // This may depend on the operator/function invoked by the node as well as its attributes.
        virtual bool Supports(const Node& node) const = 0;

        // Indicates whether the execution provider can realize/execute a given function.
        virtual bool Supports(const Function& function) const = 0;

    private:

        std::string m_id;
    };
}
#endif  // CORE_GRAPH_GRAPH_TRANSFORMER_H
