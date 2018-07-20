#pragma once

#include "core/framework/op_kernel.h"

namespace Lotus {

template <typename T>
class InstanceNorm final : public OpKernel {
 public:
  InstanceNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    LOTUS_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  float epsilon_;
};
}  // namespace Lotus
