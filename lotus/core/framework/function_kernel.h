#pragma once

#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"


namespace Lotus {
// FunctionKernel is used to provide custom implementation of functions.
// It is also used to implement custom operators.
class FunctionKernel : public OpKernel {
 public:
  explicit FunctionKernel(const OpKernelInfo& info)
      : OpKernel(info),
        provider_(info.GetExecutionProvider()){}

   Status Compute(OpKernelContext* context) const override {
    return provider_->Compute(Node(), context);
  }

 private:

  const IExecutionProvider* provider_;
};
}  // namespace Lotus
