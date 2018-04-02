#pragma once

#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"

namespace Lotus {
/* Currently there's no OpKernelInfo.provider so this doesn't work.
   It must get ignored during compilation otherwise it should break a build.

// FunctionKernel is used to provide custom implementation of functions.
// It is also used to implement custom operators.
class FunctionKernel : OpKernel {
 public:
  explicit FunctionKernel(OpKernelInfo* info)
      : node_(&info->node),
        provider_(&info->provider) {}

  void Compute(OpKernelContext* context) {
    provider_->Compute(node_, context);
  }

 private:
  LotusIR::Node* node_;
  IExecutionProvider* provider_;
};
*/
}  // namespace Lotus
