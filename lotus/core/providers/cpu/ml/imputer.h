#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
namespace ML {
class ImputerOp final : public OpKernel {
 public:
  explicit ImputerOp(const OpKernelInfo& info);
  Common::Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<float> imputed_values_float_;
  float replaced_value_float_;
  std::vector<int64_t> imputed_values_int64_;
  int64_t replaced_value_int64_;
};
}  // namespace ML
}  // namespace Lotus
