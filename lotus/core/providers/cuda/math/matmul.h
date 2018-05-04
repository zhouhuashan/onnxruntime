#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

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
