#include "core/providers/cpu/math/basic.h"

namespace Lotus{

template<typename T> auto ToEigen(      Tensor& t) { return      EigenVectorMap<T>(t.mutable_data<T>(), t.shape().Size()); }
template<typename T> auto ToEigen(const Tensor& t) { return ConstEigenVectorMap<T>(t.data<T>(),         t.shape().Size()); }

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
    
    ToEigen<float>(C) = ToEigen<float>(A).cwiseProduct(ToEigen<float>(B));
}

template<>
void Reciprocal<float>::compute(OpKernelContext* ctx) {
    auto& X = *ctx->input<Tensor>(0);
    auto& Y = *ctx->output(0, X.shape());

    ToEigen<float>(Y) = ToEigen<float>(X).cwiseInverse();
}

template<>
void Sum<float>::compute(OpKernelContext* ctx) {
    auto inputCount = node().InputArgCount().size();
    LOTUS_ENFORCE(inputCount>=1, "Must have 1 or more inputs");
    auto& data_0 = *ctx->input<Tensor>(0);
    auto& shape = data_0.shape();
    auto sum = ToEigen<float>(*ctx->output(0, shape));

    if(inputCount==1) {
       sum = ToEigen<float>(data_0);
       return;
    }

    auto& data_1 = *ctx->input<Tensor>(1);
    LOTUS_ENFORCE(data_1.shape() == shape, "All inputs must have the same shape");

    sum = ToEigen<float>(data_0) + ToEigen<float>(data_1);
    for(int index=2;index<inputCount;index++) {
        auto &data_n = *ctx->input<Tensor>(index);
        LOTUS_ENFORCE(data_n.shape() == shape, "All inputs must have the same shape");
        sum += ToEigen<float>(data_n);
    }
}

}
