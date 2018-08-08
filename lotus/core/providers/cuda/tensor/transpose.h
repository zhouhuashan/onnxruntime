#pragma once

#include "gsl/gsl_util"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace Lotus {
namespace Cuda {

template <typename T>
class Transpose final : public CudaKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : CudaKernel(info), TransposeBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace Cuda
}  // namespace Lotus
