#ifndef CORE_PROVIDERS_CPU_NO_OP_H
#define CORE_PROVIDERS_CPU_NO_OP_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

    template <typename T>
    class IdentityOp final : public OpKernel {
    public:
        IdentityOp(const OpKernelInfo& info) : OpKernel(info) {
        }

        Status compute(OpKernelContext* context) const override {
            const Tensor* X = context->input<Tensor>(0);
            const TensorShape& shape = X->shape();
            Tensor* Y = context->output(0, TensorShape(shape));
            for (size_t i = 0; i < shape.Size(); ++i) {
                Y->mutable_data<T>()[i] = X->data<T>()[i];
            }
            return Status::OK();
        }
    };

}  //namespace Lotus

#endif
