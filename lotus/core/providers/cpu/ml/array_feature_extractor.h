#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
namespace ML {
template <typename T>
class ArrayFeatureExtractorOp final : public OpKernel {
 public:
  explicit ArrayFeatureExtractorOp(const OpKernelInfo& info);
  Common::Status Compute(OpKernelContext* context) const override;
};
}  // namespace ML
}  // namespace Lotus
