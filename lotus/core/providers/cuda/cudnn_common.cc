#include "cudnn_common.h"
#include "gsl/gsl_util"
#include "shared_inc/cuda_call.h"
#include "core/providers/cpu/tensor/utils.h"

namespace Lotus {
namespace Cuda {

CudnnTensor::CudnnTensor()
    : tensor_(nullptr) {
}

CudnnTensor::~CudnnTensor() {
  if (tensor_ != nullptr) {
    cudnnDestroyTensorDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

Status CudnnTensor::Set(const std::vector<int64_t>& input_dims, cudnnDataType_t dataType) {
  if (!tensor_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&tensor_));

  int rank = gsl::narrow_cast<int>(input_dims.size());
  TensorPitches pitches(input_dims);
  std::vector<int> dims(rank + 1);
  std::vector<int> strides(pitches.size() + 1);
  for (int i = 0; i < rank; i++) {
    dims[i] = gsl::narrow_cast<int>(input_dims[i]);
    strides[i] = gsl::narrow_cast<int>(pitches[i]);
  }
  CUDNN_RETURN_IF_ERROR(cudnnSetTensorNdDescriptor(tensor_, dataType, (int)rank, dims.data(), strides.data()));
  return Status::OK();
}

template <typename ElemType>
cudnnDataType_t CudnnTensor::GetDataType() {
  if (typeid(ElemType) == typeid(float))
    return CUDNN_DATA_FLOAT;
  else if (typeid(ElemType) == typeid(double))
    return CUDNN_DATA_DOUBLE;
  else if (typeid(ElemType) == typeid(half))
    return CUDNN_DATA_HALF;
  else
    LOTUS_THROW("cuDNN engine currently supports only single/double/half precision data types.");
}

template cudnnDataType_t CudnnTensor::GetDataType<float>();
template cudnnDataType_t CudnnTensor::GetDataType<double>();
template cudnnDataType_t CudnnTensor::GetDataType<half>();

template <>
const float Consts<float>::One = 1;

template <>
const double Consts<double>::One = 1;

template <>
const float Consts<float>::Zero = 0;

template <>
const double Consts<double>::Zero = 0;

const float Consts<half>::Zero = 0;

const float Consts<half>::One = 1;

}  // namespace Cuda
}  // namespace Lotus
