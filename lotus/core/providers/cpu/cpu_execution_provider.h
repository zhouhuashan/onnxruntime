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
        CPUExecutionProvider(const ExecutionProviderInfo& info)
        {
            name_ = info.Name();
            version_ = info.Version();
            SetId();
        }

        virtual const std::string& Name() const override
        {
            return name_;
        }

        virtual const std::string& Version() const override
        {
            return version_;
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

        virtual void Compute(const Node& node, OpKernelContext* context) override
        {
            UNUSED_PARAMETER(node);
            UNUSED_PARAMETER(context);
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual Status CopyCPUTensorTo(const Tensor& srcTensor,
            Tensor* p_dstTensor) override
        {
            //no really copy needed.
            LOTUS_ENFORCE(p_dstTensor);
            p_dstTensor->ShallowCopy(srcTensor);
            return Status::OK();
        }

        virtual Status CopyTensorToCPU(const Tensor& srcTensor,
            Tensor* p_dstTensor) override
        {
            //no really copy needed.
            LOTUS_ENFORCE(p_dstTensor);
            p_dstTensor->ShallowCopy(srcTensor);
            return Status::OK();
        }

    private:
        string name_;
        string version_;
    };
}

#endif  // CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H
