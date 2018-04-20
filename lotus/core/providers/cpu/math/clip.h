#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T>
class Clip final : public OpKernel {
 public:
  Clip(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<T>("max", &max_).IsOK());
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<T>("min", &min_).IsOK());
  }

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    EigenVectorMap<T>(Y->MutableData<T>(), Y->Shape().Size()) =
        ConstEigenVectorMap<T>(X->Data<T>(), X->Shape().Size())
            .cwiseMax(min_)
            .cwiseMin(max_);
    return Status::OK();
  }

 private:
  T max_;
  T min_;
};

}  // namespace Lotus
