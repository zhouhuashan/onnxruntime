#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_base.h"

namespace Lotus {
namespace MklDnn {
template <typename T>
class Conv final : public OpKernel, public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
};
}  // namespace MklDnn
}  // namespace Lotus
