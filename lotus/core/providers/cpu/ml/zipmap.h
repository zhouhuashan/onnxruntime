#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
namespace Lotus {
namespace ML {

class ZipMapOp final : public OpKernel {
 public:
  explicit ZipMapOp(const OpKernelInfo& info);
  Common::Status Compute(OpKernelContext* context) const override;

 private:
  bool using_strings_;
  std::vector<int64_t> classlabels_int64s_;
  std::vector<std::string> classlabels_strings_;
};

}  // namespace ML
}  // namespace Lotus
