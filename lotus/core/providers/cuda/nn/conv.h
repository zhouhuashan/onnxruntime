
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_base.h"

namespace Lotus {
namespace Cuda {

template <typename T>
class Conv : public CudaKernel, public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : CudaKernel(info), ConvBase(info) {
    auto pads_size = pads_.size();
    LOTUS_ENFORCE(pads_size % 2 == 0);
    auto rank = pads_size / 2;
    for (size_t i = 0; i < rank; i++) {
      LOTUS_ENFORCE(pads_[i] == pads_[i + rank], "cudnn only supports symmetric padding");
    }
  }

  Status Compute(OpKernelContext* context) const override;
};

class CudnnFilterDescriptor final {
 public:
  CudnnFilterDescriptor();
  ~CudnnFilterDescriptor();

  Status Set(const std::vector<int64_t>& filter_dims, cudnnDataType_t data_typ);

  operator cudnnFilterDescriptor_t() const { return desc_; }

 private:
  cudnnFilterDescriptor_t desc_;
};

class CudnnConvolutionDescriptor final {
 public:
  CudnnConvolutionDescriptor();
  ~CudnnConvolutionDescriptor();

  Status Set(const std::vector<int64_t>& kernel_shape,
             const std::vector<int64_t>& pads,
             const std::vector<int64_t>& strides,
             const std::vector<int64_t>& dilations,
             cudnnConvolutionMode_t mode,
             cudnnDataType_t data_type);

  operator cudnnConvolutionDescriptor_t() const { return desc_; }

 private:
  cudnnConvolutionDescriptor_t desc_;
};

}  // namespace Cuda
}  // namespace Lotus
