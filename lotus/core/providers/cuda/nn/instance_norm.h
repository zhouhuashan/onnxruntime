#pragma once

#include "core/providers/cuda/cudnn_common.h"

namespace Lotus {
namespace Cuda {

template <typename T>
class InstanceNorm final : public CudaKernel {
 public:
  InstanceNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  double epsilon_;
};

}  // namespace Cuda
}  // namespace Lotus
