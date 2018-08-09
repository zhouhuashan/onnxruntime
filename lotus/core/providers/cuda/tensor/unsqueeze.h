#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/unsqueeze.h"

namespace Lotus {
namespace Cuda {

class Unsqueeze final : public UnsqueezeBase, public CudaKernel {
 public:
  Unsqueeze(const OpKernelInfo& info) : UnsqueezeBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace Cuda
}  // namespace Lotus
