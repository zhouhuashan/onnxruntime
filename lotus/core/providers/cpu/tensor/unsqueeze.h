#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T>
struct Unsqueeze final : OpKernel {
  Unsqueeze(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(info.GetAttrs("axes", axes_).IsOK(), "Missing/Invalid 'axes' attribute value");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> axes_;
};

}  // namespace Lotus
