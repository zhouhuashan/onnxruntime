#pragma once

#include "gsl/gsl_util"
#include "core/providers/cuda/cuda_common.h"

namespace Lotus {
namespace Cuda {

template <typename T>
class Softmax final : public CudaKernel {
 public:
  Softmax(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace Cuda
}  // namespace Lotus
