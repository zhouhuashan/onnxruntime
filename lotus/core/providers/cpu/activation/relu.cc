#include "core/providers/cpu/activation/relu.h"

namespace Lotus {

    template<>
    void ReLU<float>::compute(OpKernelContext* ctx) {
        const Tensor* X = ctx-> template input<Tensor>(0);
        Tensor* Y = ctx->output(0, X->shape());
        EigenVectorMap<float>(Y->mutable_data<float>(), Y->shape().Size()) =
            ConstEigenVectorMap<float>(X->data<float>(), X->shape().Size())
            .cwiseMax(0);
    }

}
