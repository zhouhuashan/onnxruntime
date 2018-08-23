
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_base.h"

namespace Lotus {
namespace Cuda {

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

  Status Set(size_t rank,
             const std::vector<int64_t>& pads,
             const std::vector<int64_t>& strides,
             const std::vector<int64_t>& dilations,
             cudnnConvolutionMode_t mode,
             cudnnDataType_t data_type);

  operator cudnnConvolutionDescriptor_t() const { return desc_; }

 private:
  cudnnConvolutionDescriptor_t desc_;
};

// cached cudnn descriptors
template <typename AlgoType>
struct CudnnConvState {
  // if x/w dims changed, update algo and cudnnTensors
  std::vector<int64_t> last_x_dims;
  std::vector<int64_t> last_w_dims;

  // these would be recomputed if x/w dims change
  std::vector<int64_t> y_dims;
  bool found_algo = false;
  size_t workspace_bytes = 32 * 1024 * 1024;  // initial workspace for algo search
  AlgoType algo;
  CudnnTensor x_tensor;
  CudnnFilterDescriptor filter_desc;
  CudnnTensor b_tensor;
  CudnnTensor y_tensor;
  CudnnConvolutionDescriptor conv_desc;
};

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

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable CudnnConvState<cudnnConvolutionFwdAlgo_t> s_;
};

}  // namespace Cuda
}  // namespace Lotus
