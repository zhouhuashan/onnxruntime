#include "core/kernels/cpu/math/clip.h"

namespace Lotus{

template<>
void Clip<float>::Compute(OpKernelContext* ctx) {
    const Tensor* X = ctx-> template Input<Tensor>(0);
    Tensor* Y = ctx->template Output<Tensor>(0);
    auto Y_ptr = TensorUtil::ReshapeTensor(*Y, X->shape());
    Y = Y_ptr.get();
    EigenVectorMap<float>(Y_ptr->mutable_data<float>(), Y_ptr->shape().Size()) =
        ConstEigenVectorMap<float>(X->data<float>(), X->shape().Size())
        .cwiseMax(min_)
        .cwiseMin(max_);
}

}
