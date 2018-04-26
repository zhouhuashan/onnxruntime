#pragma once
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
class FC final : public OpKernel {
 public:
  FC(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("axis_w", &axis_w_).IsOK()) {
      axis_w_ = 1;
    }
  }

  Status Compute(OpKernelContext* context) const override {
    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);

    int64_t M, K, N;
    M = X->Shape().SizeToDimension(axis_);
    K = X->Shape().SizeFromDimension(axis_);
    N = W->Shape().SizeToDimension(axis_w_);

    //dimension check
    if ((M != X->Shape().Size() / K) ||
        (K != W->Shape().Size() / N) ||
        (B->Shape().NumDimensions() != 1) ||
        (B->Shape().Size() != N)) {
      return Status(LOTUS, INVALID_ARGUMENT,
                    "FC: Dimension mismatch. X: " + X->Shape().ToString() +
                        ", W: " + W->Shape().ToString() +
                        ", B: " + B->Shape().ToString() +
                        ", axis: " + std::to_string(axis_) +
                        ", axis_w: " + std::to_string(axis_w_));
    }

    if (B->Shape().NumDimensions() != 1) {
      return Status(LOTUS, INVALID_ARGUMENT, "FC: Invalid bias with dimension " + std::to_string(B->Shape().NumDimensions()));
    }

    auto Y = context->Output(0, TensorShape(std::vector<int64_t>{M, N}));

    //bias
    // Todo: we might should move this part into math::gemm to let eigen
    // have better chance to further optimize it.
    auto output_mat = EigenMatrixMapRowMajor<T_Y>(
        Y->template MutableData<T_Y>(),
        M,
        N);
    output_mat.setZero();
    auto bias_vec = ConstEigenVectorMap<T_B>(
        B->template Data<T_B>(),
        N);
    output_mat.rowwise() += bias_vec.transpose();

    // W * x
    Math::Gemm<T_X, CPUMathUtil>(
        CblasNoTrans,
        CblasTrans,
        M,
        N,
        K,
        1,
        X->template Data<T_X>(),
        W->template Data<T_W>(),
        1,
        Y->template MutableData<T_Y>(),
        &CPUMathUtil::Instance());

    return Status::OK();
  }

 private:
  int64_t axis_;
  int64_t axis_w_;
};  // namespace Lotus

}  // namespace Lotus
