#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

struct Constant final : OpKernel {
  Constant(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(info.GetAttr("value", &value_).IsOK(), "Must have valid 'value' attribute");
  }
  Status Compute(OpKernelContext* context) const override;

 private:
  TensorProto value_;
};

}  // namespace Lotus
