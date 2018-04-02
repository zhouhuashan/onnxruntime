#pragma once

#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {
template <typename T>
class Softmax final : public OpKernel {
 public:
  Softmax(const OpKernelInfo& info) : OpKernel{info}, axis_{1} {
    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    }

    // if value was provided, make sure it was valid
    LOTUS_ENFORCE(axis_ >= 0, "Invalid axis provided.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int axis_;
};
}  // namespace Lotus
