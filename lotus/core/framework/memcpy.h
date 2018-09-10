#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  const IExecutionProvider* provider_;
};

}  // namespace onnxruntime
