#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"

namespace Lotus {

class UnsqueezeBase {
 protected:
  UnsqueezeBase(const OpKernelInfo& info) {
    LOTUS_ENFORCE(info.GetAttrs("axes", axes_).IsOK(), "Missing/Invalid 'axes' attribute value");
  }

  struct Prepare {
    const Tensor* input_tensor;
    Tensor* output_tensor;
  };

  Status PrepareCompute(OpKernelContext* context, Prepare& p) const;

 private:
  std::vector<int64_t> axes_;
};

class Unsqueeze final : public OpKernel, public UnsqueezeBase {
 public:
  Unsqueeze(const OpKernelInfo& info) : UnsqueezeBase(info), OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace Lotus
