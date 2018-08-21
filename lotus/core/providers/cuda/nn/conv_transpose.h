#pragma once

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_transpose.h"

namespace Lotus {
namespace Cuda {

template <typename T>
class ConvTranspose : public CudaKernel, public ConvTransposeBase {
 public:
  ConvTranspose(const OpKernelInfo& info) : CudaKernel(info), ConvTransposeBase(info){};
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // cudnn algorithm search is slow, so cache it with input shape
  // the map is mutable because Compute is const
  mutable std::map<std::vector<int64_t>, cudnnConvolutionBwdDataAlgo_t> algo_cache_;
};

}  // namespace Cuda
}  // namespace Lotus
