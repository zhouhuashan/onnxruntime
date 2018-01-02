#ifndef CORE_GRAPH_GRAPH_TRANSFORMER_H
#define CORE_GRAPH_GRAPH_TRANSFORMER_H

#include "core/graph/graph.h"

namespace LotusIR
{
    // A graph transformer interface. A graph transformer could be
    // going thru a graph to do some optimization, for example, op fusion.
    class IGraphTransformer
    {
    public:

        virtual ~IGraphTransformer() {}

        // Transformer name.
        virtual const std::string& Name() const = 0;

        // Transformer description.
        virtual const std::string& Description() const {
            return "";
        }

        // Apply <*this> transformation to a specific graph.
        // Transformation happens in place.
        // The return value of "modified" indicates if the graph was modified or not.
        virtual Status Apply(/*IN/OUT*/ Graph& p_graph, /*OUT*/ bool& modified) = 0;
    };

    class GraphTransformerManager
    {
    public:

        // Register a graph transformer.
        Status Register(const IGraphTransformer& p_graphTransformer);

        // Going thru all transformers registered in <*this> manager on specified graph.
        Status ApplyAll(/*IN/OUT*/ Graph& p_graph);

        static GraphTransformerManager Instance()
        {
            static GraphTransformerManager s_graphProcessorRegistry;
            return s_graphProcessorRegistry;
        }

    private:

        GraphTransformerManager() = default;

        std::vector<IGraphTransformer> m_transformers;
    };

#define REGISTER_GRAPH_PROCESSOR(ProcessorClassName) REGISTER_GRAPH_PROCESSOR_UNIQ_HELPER(__COUNTER__, ProcessorClassName)
#define REGISTER_GRAPH_PROCESSOR_UNIQ_HELPER(Counter, ProcessorClassName) REGISTER_GRAPH_PROCESSOR_UNIQ(Counter, ProcessorClassName)
#define REGISTER_GRAPH_PROCESSOR_UNIQ(Counter, ProcessorClassName)          \
    static Status status_##Counter                                          \
    = GraphTransformerManager::Instance().Register(ProcessorClassName());

    // Example
    class A : public IGraphTransformer {
    public:

        virtual const std::string& Name() const override
        {
            return "A";
        }

        virtual Status Apply(/*IN/OUT*/ Graph& p_graph, /*OUT*/ bool& modified) override
        {
            modified = false;
            return Status::OK();
        }
    };
    REGISTER_GRAPH_PROCESSOR(A);


    // Function representation class.
    class Function : public GraphBase
    {
    public:

        // Get <*this> function's schema.
        const OperatorSchema& GetSchema() const;

    private:

        OperatorSchema m_schema;
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
