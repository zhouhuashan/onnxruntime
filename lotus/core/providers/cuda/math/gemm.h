#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace Lotus {
namespace Cuda {
template <typename T>
class Gemm final : public CudaKernel {
  using Base = CudaKernel;

 public:
  Gemm(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t temp;
    LOTUS_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    LOTUS_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    LOTUS_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    LOTUS_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;
};
}  // namespace Cuda
}  // namespace Lotus
