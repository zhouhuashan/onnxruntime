#ifndef CORE_FRAMEWORK_FUNCTION_KERNEL_H
#define CORE_FRAMEWORK_FUNCTION_KERNEL_H

#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"

namespace Lotus {
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
  Node* m_node;
  ExecutionProvider* m_provider;
};
}  // namespace Lotus
#endif  // CORE_FRAMEWORK_FUNCTION_KERNEL_H
