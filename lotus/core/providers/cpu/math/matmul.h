#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

template <typename T>
class MatMul final : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info)
      : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

}  // namespace Lotus
