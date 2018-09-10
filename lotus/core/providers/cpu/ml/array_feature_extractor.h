#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class ArrayFeatureExtractorOp final : public OpKernel {
 public:
  explicit ArrayFeatureExtractorOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;
};
}  // namespace ml
}  // namespace onnxruntime
