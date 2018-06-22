#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "../cuda_common.h"

namespace Lotus {
namespace Cuda {
template <typename T>
struct Tile final : CudaKernel {
  Tile(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
};
}  // namespace Cuda
}  // namespace Lotus
