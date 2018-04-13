#pragma once

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
  template <typename T>
  using EigenMatrixMapRowMajor = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  Gemm(const OpKernelInfo& info) : OpKernel(info) {
    int64_t temp;
    LOTUS_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;

    LOTUS_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    LOTUS_ENFORCE(info.GetAttr<int64_t>("broadcast", &broadcast_).IsOK());
    LOTUS_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    LOTUS_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);
    //dimension check
    LOTUS_ENFORCE(X->Shape().NumDimensions() == 2);
    LOTUS_ENFORCE(W->Shape().NumDimensions() == 2);
    // batch size
    int64_t M, K, N;

    if (trans_A_ == CblasTrans) {
      M = X->Shape()[1];
      K = X->Shape()[0];
    } else {
      M = X->Shape()[0];
      K = X->Shape()[1];
    }

    if (trans_B_ == CblasTrans) {
      N = W->Shape()[0];
    } else {
      N = W->Shape()[1];
    }

    LOTUS_ENFORCE(W->Shape().Size(), K * N);

    if (broadcast_) {
      LOTUS_ENFORCE(B->Shape().NumDimensions() == 1);
      LOTUS_ENFORCE(B->Shape()[0] == N);
    } else {
      LOTUS_ENFORCE(B->Shape().NumDimensions() == 2);
      LOTUS_ENFORCE(B->Shape()[0] == M && B->Shape()[1] == N);
    }

    LOTUS_ENFORCE(M > 0 && N > 0 && K > 0);
    auto Y = context->Output(0, TensorShape(std::vector<int64_t>{M, N}));

    //bias
    // Todo: we might should move this part into math::gemm to let eigen
    // have better chance to further optimize it.
    if (beta_ != 0) {
      auto output_mat = EigenMatrixMapRowMajor<T_Y>(
          Y->template MutableData<T_Y>(),
          M,
          N);
      output_mat.setZero();

      if (broadcast_) {
        auto bias_vec = ConstEigenVectorMap<T_B>(
            B->template Data<float>(),
            N);
        output_mat.rowwise() += bias_vec.transpose();
      } else {
        auto bias_mat = ConstEigenMatrixMap<T_B>(
            B->template Data<T_B>(),
            M,
            N);
        output_mat += bias_mat;
      }
    }

    // W * x
    Math::Gemm<T_X, CPUMathUtil>(
        trans_A_,
        trans_B_,
        M,
        N,
        K,
        alpha_,
        X->template Data<T_X>(),
        W->template Data<T_W>(),
        beta_,
        Y->template MutableData<T_Y>(),
        &CPUMathUtil::Instance());

    return Status::OK();
  }

 private:
  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  int64_t broadcast_;
  float alpha_;
  float beta_;
};

}  // namespace Lotus
