#pragma once

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_transpose.h"
#include "conv.h"

namespace Lotus {
namespace Cuda {

template <typename T>
class ConvTranspose : public CudaKernel, public ConvTransposeBase {
 public:
  ConvTranspose(const OpKernelInfo& info) : CudaKernel(info), ConvTransposeBase(info){};
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable CudnnConvState<cudnnConvolutionBwdDataAlgo_t> s_;
};

}  // namespace Cuda
}  // namespace Lotus
