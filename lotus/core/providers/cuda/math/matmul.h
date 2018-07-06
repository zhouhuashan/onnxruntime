#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace Lotus {
namespace Cuda {
template <typename T>
class MatMul final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMul(const OpKernelInfo& info)
      : CudaKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace Cuda
}  // namespace Lotus
