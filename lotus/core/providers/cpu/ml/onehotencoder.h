#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
namespace ML {
template <typename T>
class OneHotEncoderOp final : public OpKernel {
 public:
  explicit OneHotEncoderOp(const OpKernelInfo& info);
  Common::Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> InferOutputSize(const TensorShape& input_shape) const;

 private:
  std::unordered_map<int64_t, size_t> cats_int64s_;
  std::unordered_map<std::string, size_t> cats_strings_;
  int64_t zeros_;
  int64_t num_category_;
};
}  // namespace ML
}  // namespace Lotus
