#include "core/providers/cpu/activation/tanh.h"

namespace Lotus {

    template<>
    void Tanh<float>::compute(OpKernelContext* ctx) {
        const Tensor* X = ctx-> template input<Tensor>(0);
        Tensor* Y = ctx->output(0, X->shape());
        
        ConstEigenVectorArrayMap<float> xM(X->data<float>(), X->shape().Size());
        EigenVectorMap<float>(Y->mutable_data<float>(), Y->shape().Size()) = 1 - 2 * ((xM * 2).exp() + 1).inverse();
    }

}