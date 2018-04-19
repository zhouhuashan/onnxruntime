#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
namespace ML {
template <typename T>
class BinarizerOp final : public OpKernel {
 public:
  explicit BinarizerOp(const OpKernelInfo& info);
  Common::Status Compute(OpKernelContext* context) const override;

 private:
  float threshold_ = 0.f;
};
}  // namespace ML
}  // namespace Lotus
