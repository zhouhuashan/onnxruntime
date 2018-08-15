#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/squeeze.h"

namespace Lotus {
namespace Cuda {

class Squeeze final : public SqueezeBase, public CudaKernel {
 public:
  Squeeze(const OpKernelInfo& info) : SqueezeBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace Cuda
}  // namespace Lotus
