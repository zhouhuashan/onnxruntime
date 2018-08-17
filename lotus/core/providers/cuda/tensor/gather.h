#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/gather.h"

namespace Lotus {
namespace Cuda {

class Gather final : public CudaKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : GatherBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace Cuda
}  // namespace Lotus
