#ifndef CORE_FRAMEWORK_EXECUTION_PROVIDER_H
#define CORE_FRAMEWORK_EXECUTION_PROVIDER_H

#include "core/framework/allocator.h"
#include "core/framework/workspace.h"
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"

using namespace LotusIR;

namespace Lotus
{
    

    // Logical device represenatation.
    // 
    class IExecutionProvider
    {
    public:

        IExecutionProvider()
        {
            m_id = Name() + "." + Version();
        }

        virtual ~IExecutionProvider() {}

        virtual const std::string& Name() const = 0;

        virtual const std::string& Version() const = 0;

        virtual const std::string& ID() const
        {
            return m_id;
        }

        // Get graph transformer related to thsi execution provider so that it will do in-place graph
        // modeification (fusion etc) to identify which nodes will be run against <*this> execution provider.
        virtual IGraphTransformer& GetTransformer() const = 0;

        // Execute the <p_node> given <p_workSpace> which contains inputs/outputs for <*this> execution provider.
        virtual Status Execute(const Node& p_node, WorkSpace& p_workSpace) = 0;

        virtual Status CopyCPUTensorTo(const Tensor* p_srcTensor,
            Tensor* p_dstTensor) = 0;

        virtual Status CopyTensorToCPU(const Tensor* p_srcTensor,
            Tensor* p_dstTensor) = 0;

    private:

        std::string m_id;
    };

    typedef std::function<std::vector<IExecutionProvider>()> ExecutionProviderFinder;
    // Singleton execution provider manager.
    // It holds a global provider type to provider finder map, and will find/create
    // execution provider instances for inference engine.
    class ExecutionProviderMgr
    {
    public:

        static ExecutionProviderMgr Instance()
        {
            static ExecutionProviderMgr s_providerMgr;
            return s_providerMgr;
        }

        // TODO: registration for provider type to provider finder.

    private:

        ExecutionProviderMgr() {}

        std::unordered_map<std::string, ExecutionProviderFinder> m_executionProviderTypeToFinder;
    };
}
#endif  // CORE_FRAMEWORK_EXECUTION_PROVIDER_H
