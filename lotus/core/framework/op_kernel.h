#ifndef CORE_FRAMEWORK_OP_KERNEL_H
#define CORE_FRAMEWORK_OP_KERNEL_H

#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/framework/execution_frame.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"

namespace Lotus {
    class OpKernelContext;

    class OpKernelInfo
    {
    public:
        explicit OpKernelInfo(const LotusIR::Node& node,
            const AllocatorInfo& allocator_info)
            : node_(node),
            allocator_info_(allocator_info) {}

        //Get a single attribute
        template<typename T>
        Status GetAttr(const std::string& name, T* value) const;

        //Get repeated attributes
        template<typename T>
        Status GetAttrs(const std::string& name, std::vector<T>& values) const;

        const LotusIR::Node& node() const {
            return node_;
        }

        const AllocatorInfo& get_allocator_info() const {
            return allocator_info_;
        }

    private:

        const LotusIR::Node& node_;
        const AllocatorInfo& allocator_info_;
    };

    class OpKernel {
    public:
        typedef std::function<void()> DoneCallback;

        explicit OpKernel(OpKernelInfo* info)
            : op_kernel_info_(info),
            allocator_info_(info->get_allocator_info()) {
            LOTUS_ENFORCE(nullptr != info);
        }

        const LotusIR::Node& node() const {
            return op_kernel_info_->node();
        }

        virtual void compute(OpKernelContext* context) = 0;
        virtual void compute_async(OpKernelContext* context, DoneCallback done) {
            UNUSED_PARAMETER(context);
            UNUSED_PARAMETER(done);
            LOTUS_NOT_IMPLEMENTED;
        }

        const AllocatorInfo& allocator() { return allocator_info_; }

    private:

        const AllocatorInfo& allocator_info_;

        OpKernelInfo* op_kernel_info_;
    };

    class OpKernelContext {
    public:
        typedef std::unordered_map<std::string, size_t> ArgMap;

        explicit OpKernelContext(ExecutionFrame* frame, OpKernel* kernel)
            : execution_frame_(frame),
            kernel_(kernel)
        {
            LOTUS_ENFORCE(nullptr != frame && kernel != nullptr);
            arg_start_index_ = frame->get_first_arg_index(kernel->node().Index());
        }

        ~OpKernelContext() {};

        template<typename T>
        const T* input(int index) const {
            return execution_frame_->get_input<T>(arg_start_index_ + index);
        }

        template<typename T>
        T* output(int index) {
            auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->node().InputDefs().size()) + index;
            return execution_frame_->get_output<T>(output_arg_index);
        }

        // In the case that memory allocation has not been done for an output tensor,
        // The memory allocation will be done on-the-fly with given tensor shape.
        Tensor* output(int index, const TensorShape& shape);

    private:
        ExecutionFrame* execution_frame_ = nullptr;

        OpKernel* kernel_ = nullptr;

        // The argument starting index in ExecutionFrame.
        int arg_start_index_ = -1;
    };
}
#endif  // CORE_FRAMEWORK_OP_KERNEL_H
