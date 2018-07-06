#pragma once
#include "cuda_common.h"
#include "core/framework/tensor.h"

namespace Lotus {
namespace Cuda {

class CudnnTensor final {
 public:
  CudnnTensor();
  ~CudnnTensor();

  Status Set(const std::vector<int64_t>& input_dims, cudnnDataType_t dataType);

  operator cudnnTensorDescriptor_t() const { return tensor_; }

  template <typename T>
  static cudnnDataType_t GetDataType();

 private:
  cudnnTensorDescriptor_t tensor_;
};

template <typename ElemType>
struct Consts {
  static const ElemType Zero;
  static const ElemType One;
};

template <>
struct Consts<half> {
  static const float Zero;
  static const float One;
};

class CudnnReduceDescriptor final {
 public:
  CudnnReduceDescriptor();
  ~CudnnReduceDescriptor();

  Status Set(cudnnReduceTensorOp_t op, cudnnDataType_t type, cudnnReduceTensorIndices_t indices);

  operator cudnnReduceTensorDescriptor_t() const { return desc_; }

 private:
  cudnnReduceTensorDescriptor_t desc_;
};

}  // namespace Cuda
}  // namespace Lotus
