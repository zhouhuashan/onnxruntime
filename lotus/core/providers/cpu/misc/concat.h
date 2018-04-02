#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T>
struct Concat final : OpKernel {
  Concat(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr("axis", &axis_).IsOK()) {
      LOTUS_ENFORCE(false, "Must have valid 'axis' attribute");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace Lotus
