#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/pad.h"

namespace Lotus {
namespace Cuda {

template <typename T>
class Pad final : public PadBase, public CudaKernel {
 public:
  Pad(const OpKernelInfo& info) : PadBase(info), CudaKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace Cuda
}  // namespace Lotus
