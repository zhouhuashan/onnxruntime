#include "core/kernels/cpu/math/add.h"

namespace Lotus{

template<typename T>      EigenVectorMap<T> ToEigen(      Tensor& t) { return      EigenVectorMap<T>(t.mutable_data<T>(), t.shape().Size()); }
template<typename T> ConstEigenVectorMap<T> ToEigen(const Tensor& t) { return ConstEigenVectorMap<T>(t.data<T>(),         t.shape().Size()); }

template<>
void Constant<float>::compute(OpKernelContext* ctx) {
    auto& C = *ctx->output(0, value_->shape());
    C;
//    ToEigen<float>(C) = ToEigen<float>(*value_);
}

template<>
void Concat<float>::compute(OpKernelContext* ctx) {
    auto inputCount = node().InputArgCount().size();
    LOTUS_ENFORCE(inputCount>=1, "Must have 1 or more inputs");

    auto& data_0 = *ctx->input<Tensor>(0);
    auto& shape = data_0.shape();
    auto sum = ToEigen<float>(*ctx->output(0, shape));

    if(inputCount==1)
    {
       sum = ToEigen<float>(data_0);
       return;
    }


}

template<>
void Add<float>::compute(OpKernelContext* ctx) {
    auto& A = *ctx->input<Tensor>(0);
    auto& B = *ctx->input<Tensor>(1);
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    auto& C = *ctx->output(0, A.shape());

    ToEigen<float>(C) = ToEigen<float>(A) + ToEigen<float>(B);
}

template<>
void Sub<float>::compute(OpKernelContext* ctx) {
    auto& A = *ctx->input<Tensor>(0);
    auto& B = *ctx->input<Tensor>(1);
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    auto& C = *ctx->output(0, A.shape());

    ToEigen<float>(C) = ToEigen<float>(A) - ToEigen<float>(B);
}

template<>
void Mul<float>::compute(OpKernelContext* ctx) {
    auto& A = *ctx->input<Tensor>(0);
    auto& B = *ctx->input<Tensor>(1);
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    auto& C = *ctx->output(0, A.shape());

    ToEigen<float>(C) = ToEigen<float>(A) * ToEigen<float>(B);
}

template<>
void Sum<float>::compute(OpKernelContext* ctx) {
    auto inputCount = node().InputArgCount().size();
    LOTUS_ENFORCE(inputCount>=1, "Must have 1 or more inputs");
    auto& data_0 = *ctx->input<Tensor>(0);
    auto& shape = data_0.shape();
    auto sum = ToEigen<float>(*ctx->output(0, shape));

    if(inputCount==1)
    {
       sum = ToEigen<float>(data_0);
       return;
    }

    auto& data_1 = *ctx->input<Tensor>(1);
    LOTUS_ENFORCE(data_1.shape() == shape, "All inputs must have the same shape");

    sum = ToEigen<float>(data_0) + ToEigen<float>(data_1);
    for(int index=2;index<inputCount;index++)
    {
        auto &data_n = *ctx->input<Tensor>(index);
        LOTUS_ENFORCE(data_n.shape() == shape, "All inputs must have the same shape");
        sum += ToEigen<float>(data_n);
    }
}

}
