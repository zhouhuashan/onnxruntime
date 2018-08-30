#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"

namespace Lotus {
namespace MklDnn {
template <typename T>
class Conv final : public Lotus::Conv<T> {
 public:
  Conv(const OpKernelInfo& info) : Lotus::Conv<T>(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
};
}  // namespace MklDnn
}  // namespace Lotus
