#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace Lotus {
namespace Cuda {

class Flatten final : public CudaKernel {
 public:
  Flatten(const OpKernelInfo& info) : CudaKernel(info) {
    LOTUS_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace Cuda
}  // namespace Lotus
