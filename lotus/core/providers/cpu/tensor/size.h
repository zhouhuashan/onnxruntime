#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Size final : public OpKernel {
 public:
  Size(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
