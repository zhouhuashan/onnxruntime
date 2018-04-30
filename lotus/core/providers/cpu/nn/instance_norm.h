#pragma once

#include "core/framework/op_kernel.h"

namespace Lotus {

template <typename T>
class InstanceNorm final : public OpKernel {
 public:
  InstanceNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    op_kernel_info.GetAttr<float>("epsilon", &epsilon_);
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  float epsilon_ = 1e-5f;
};
}  // namespace Lotus
