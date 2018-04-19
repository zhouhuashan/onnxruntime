#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
namespace ML {
template <typename T>
class ScalerOp final : public OpKernel {
 public:
  explicit ScalerOp(const OpKernelInfo& info);
  Common::Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<float> scale_;
  std::vector<float> offset_;
};
}  // namespace ML
}  // namespace Lotus
