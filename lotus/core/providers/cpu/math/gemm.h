#ifndef CORE_PROVIDERS_CPU_MATH_GEMM_H
#define CORE_PROVIDERS_CPU_MATH_GEMM_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T_X,
          typename T_W,
          typename T_B,
          typename T_Y>
class Gemm final : public OpKernel {
 public:
  Gemm(const OpKernelInfo& info) : OpKernel(info) {
    int64_t temp;
    LOTUS_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    transA_ = temp == 0 ? CblasNoTrans : CblasTrans;

    LOTUS_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    transB_ = temp == 0 ? CblasNoTrans : CblasTrans;

    LOTUS_ENFORCE(info.GetAttr<int64_t>("broadcast", &broadcast_).IsOK());
    LOTUS_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    LOTUS_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  void compute(OpKernelContext* context) override {
    const auto X = context->input<Tensor>(0);
    const auto W = context->input<Tensor>(1);
    const auto B = context->input<Tensor>(2);
    //dimension check
    LOTUS_ENFORCE(X->shape().NumDimensions() == 2);
    LOTUS_ENFORCE(W->shape().NumDimensions() == 2);
    // batch size
    int64_t M, K, N;
    if (transA_ == CblasTrans) {
      M = X->shape()[1];
      K = X->shape()[0];
    } else {
      M = X->shape()[0];
      K = X->shape()[1];
    }
    if (transB_ == CblasTrans) {
      N = W->shape()[0];
    } else {
      N = W->shape()[1];
    }
    LOTUS_ENFORCE(W->shape().Size(), K * N);
    if (broadcast_) {
      LOTUS_ENFORCE(B->shape().NumDimensions() == 1);
      LOTUS_ENFORCE(B->shape()[0] == N);
    } else {
      LOTUS_ENFORCE(B->shape().NumDimensions() == 2);
      LOTUS_ENFORCE(B->shape()[0] == M && B->shape()[1] == N);
    }

    LOTUS_ENFORCE(M > 0 && N > 0 && K > 0);
    auto Y = context->output(0, TensorShape(std::vector<int64_t>{M, N}));

    //bias
    // Todo: we might should move this part into math::gemm to let eigen
    // have better chance to further optimize it.
    if (beta_ != 0) {
      auto output_mat = EigenMatrixMap<T_Y>(
          Y->template mutable_data<T_Y>(),
          M,
          N);
      output_mat.setZero();
      if (broadcast_) {
        auto bias_vec = ConstEigenVectorMap<T_B>(
            B->template data<float>(),
            N);
        output_mat.rowwise() += bias_vec.transpose();
      } else {
        auto bias_mat = ConstEigenMatrixMap<T_B>(
            B->template data<T_B>(),
            M,
            N);
        output_mat += bias_mat;
      }
    }

    // W * x
    math::Gemm<T_X, CPUMathUtil>(
        transA_,
        transB_,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        alpha_,
        X->template data<T_X>(),
        W->template data<T_W>(),
        beta_,
        Y->template mutable_data<T_Y>(),
        &CPUMathUtil::Instance());
  }

 private:
  CBLAS_TRANSPOSE transA_;
  CBLAS_TRANSPOSE transB_;
  int64_t broadcast_;
  float alpha_;
  float beta_;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_MATH_CLIP_H
