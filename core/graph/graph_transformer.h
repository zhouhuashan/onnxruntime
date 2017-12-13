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


    class IExecutionProvider
    {
    public:

        virtual ~IExecutionProvider() {}

        virtual const std::string& ID() const = 0;

		// There is a tradeoff between the following two API methods for declaring an execution-provider's capabilities.
		// The first is more general (e.g., if an execution-provider supports an operation with specific attribute-values).
		// The second could be a more efficient approximation that is useful in some contexts: for example, for a
		// graph-transformer that recognizes all occurrences of sub-graphs that can be replaced by a function call.

		// Indicates whether the execution provider can realize/execute a given node.
		// This may depend on the operator/function invoked by the node as well as its attributes.
		virtual bool Supports(const Node& node) const = 0;

		// Indicates whether the execution provider can realize/execute a given function.
		virtual bool Supports(const Function& function) const = 0;
    };
}
#endif  // CORE_GRAPH_GRAPH_TRANSFORMER_H
