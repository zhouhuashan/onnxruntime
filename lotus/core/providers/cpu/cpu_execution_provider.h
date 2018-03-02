#ifndef CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H
#define CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H

#include "core/framework/execution_provider.h"
#include "core/framework/allocatormgr.h"

namespace Lotus
{
    // Logical device represenatation.
    class CPUExecutionProvider : public IExecutionProvider
    {
    public:
        CPUExecutionProvider()
        {
            //Todo: implement name and version
            //SetId();
        }

        virtual const std::string& Name() const override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual const std::string& Version() const override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual IGraphTransformer& GetTransformer() const override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual IArenaAllocator& GetAllocator() const override
        {
            auto alloc_mgr = AllocatorManager::Instance();
            LOTUS_ENFORCE(alloc_mgr);
            return alloc_mgr->GetArena(CPU);
        }

        virtual void Compute(Node* node, OpKernelContext* context) override
        {
            UNUSED_PARAMETER(node);
            UNUSED_PARAMETER(context);
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual Status CopyCPUTensorTo(const Tensor& srcTensor,
            Tensor* p_dstTensor) override
        {
            UNUSED_PARAMETER(srcTensor);
            UNUSED_PARAMETER(p_dstTensor);
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual Status CopyTensorToCPU(const Tensor& srcTensor,
            Tensor* p_dstTensor) override
        {
            UNUSED_PARAMETER(srcTensor);
            UNUSED_PARAMETER(p_dstTensor);
            LOTUS_NOT_IMPLEMENTED;
        }

    private:
    };
}

#endif  // CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H
