#ifndef CORE_GRAPH_GRAPH_TRANSFORMER_H
#define CORE_GRAPH_GRAPH_TRANSFORMER_H

#include "core/graph/graph.h"

namespace LotusIR
{
    // A graph transformer interface. A graph transformer could be
    // going thru a graph to do some optimization, for example, op fusion.
    class IGraphTransoformer
    {
    public:

        virtual ~IGraphTransoformer() {}

        // Transformer name.
        virtual const std::string& Name() const = 0;

        // Transformer description.
        virtual const std::string& Description() const {
            return "";
        }

        // Apply <*this> transformation to a specific graph.
        // Transformation happens in place.
        virtual Status Apply(/*IN/OUT*/ Graph& p_graph) = 0;
    };

    class GraphTransformerManager
    {
    public:

        // Register a graph transformer.
        Status Register(const IGraphTransoformer& p_graphTransformer);

        // Going thru all transformers registered in <*this> manager on specified graph.
        Status ApplyAll(/*IN/OUT*/ Graph& p_graph);

        static GraphTransformerManager Instance()
        {
            static GraphTransformerManager s_graphProcessorRegistry;
            return s_graphProcessorRegistry;
        }

    private:

        GraphTransformerManager() = default;

        std::vector<IGraphTransoformer> m_transformers;
    };

#define REGISTER_GRAPH_PROCESSOR(ProcessorClassName) REGISTER_GRAPH_PROCESSOR_UNIQ_HELPER(__COUNTER__, ProcessorClassName)
#define REGISTER_GRAPH_PROCESSOR_UNIQ_HELPER(Counter, ProcessorClassName) REGISTER_GRAPH_PROCESSOR_UNIQ(Counter, ProcessorClassName)
#define REGISTER_GRAPH_PROCESSOR_UNIQ(Counter, ProcessorClassName)          \
    static Status status_##Counter                                          \
    = GraphTransformerManager::Instance().Register(ProcessorClassName());

    // Example
    class A : public IGraphTransoformer {
    public:

        virtual const std::string& Name() const override
        {
            return "A";
        }

        virtual Status Apply(/*IN/OUT*/ Graph& p_graph) override
        {
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




    };
}
#endif  // CORE_GRAPH_GRAPH_TRANSFORMER_H
