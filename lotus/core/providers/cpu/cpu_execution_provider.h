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
        explicit CPUExecutionProvider(const ExecutionProviderInfo& /*info*/) {}

        virtual IGraphTransformer& GetTransformer() const override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual IArenaAllocator& GetTempSpaceAllocator() const override
        {
            auto alloc_mgr = AllocatorManager::Instance();
            LOTUS_ENFORCE(alloc_mgr);
            return alloc_mgr->GetArena(CPU);
        }

        virtual void Compute(const Node& node, OpKernelContext* context) override
        {
            UNUSED_PARAMETER(node);
            UNUSED_PARAMETER(context);
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual Status CopyTensorTo(const Tensor& srcTensor,
            Tensor* p_dstTensor) override
        {
            LOTUS_ENFORCE(p_dstTensor && p_dstTensor->location().name_ == CPU);
            // Todo: support copy with different devices.
            if (srcTensor.location().name_ != CPU)
                LOTUS_NOT_IMPLEMENTED;
            //no really copy needed if is copy to cpu.
            p_dstTensor->ShallowCopy(srcTensor);
            return Status::OK();
        }

        virtual Status CopyTensorFrom(const Tensor& srcTensor,
            Tensor* p_dstTensor) override
        {
            LOTUS_ENFORCE(p_dstTensor && srcTensor.location().name_ == CPU);
            // Todo: support copy with different devices.
            if (p_dstTensor->location().name_ != CPU)
                LOTUS_NOT_IMPLEMENTED;
            //no really copy needed.
            p_dstTensor->ShallowCopy(srcTensor);
            return Status::OK();
        }
    };
}

#endif  // CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H
