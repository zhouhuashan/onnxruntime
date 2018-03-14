#include "core/providers/cpu/activation/sigmoid.h"

namespace Lotus {

    template<>
    void Sigmoid<float>::compute(OpKernelContext* ctx) {
        const Tensor* X = ctx-> template input<Tensor>(0);
        Tensor* Y = ctx->output(0, X->shape());

        ConstEigenVectorArrayMap<float> xM(X->data<float>(), X->shape().Size());
        EigenVectorArrayMap<float>(Y->mutable_data<float>(), Y->shape().Size()) = 1. / (1. + (-xM).exp());
    }

}