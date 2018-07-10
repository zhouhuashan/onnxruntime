
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/nn/pool.h"

namespace Lotus {
namespace Cuda {

#define POOLING_KERNEL(op_name, data_type, pool_type)                              \
  REGISTER_KERNEL(KernelDefBuilder(op_name)                                        \
                    .Domain(LotusIR::kOnnxDomain)                                  \
                    .SinceVersion(1)                                               \
                    .Provider(LotusIR::kCudaExecutionProvider)                     \
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),\
                Pool<data_type, pool_type>);

POOLING_KERNEL("AveragePool", float, AveragePool)
POOLING_KERNEL("AveragePool", double, AveragePool)
POOLING_KERNEL("GlobalAveragePool", float, AveragePool)
POOLING_KERNEL("GlobalAveragePool", double, AveragePool)
POOLING_KERNEL("MaxPool", float, MaxPool)
POOLING_KERNEL("MaxPool", double, MaxPool)
POOLING_KERNEL("GlobalMaxPool", float, MaxPool)
POOLING_KERNEL("GlobalMaxPool", double, MaxPool)

template <typename T, PoolType type>
Status Pool<T, type>::Compute(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "Input dimension cannot be less than 3.");
  }

  std::vector<int64_t> kernel_shape = kernel_shape_;
  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> strides = strides_;

  if (global_pooling_) {
    kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    pads.assign(kernel_shape.size(), 0);
    strides.assign(kernel_shape.size(), 1);
  }

  std::vector<int64_t> y_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor x_tensor;
  CudnnTensor y_tensor;
  LOTUS_RETURN_IF_ERROR(x_tensor.Set(x_dims, CudnnTensor::GetDataType<CudaT>()));
  LOTUS_RETURN_IF_ERROR(y_tensor.Set(y_dims, CudnnTensor::GetDataType<CudaT>()));

  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  if (type == Lotus::Cuda::PoolType::AveragePool) {
    mode = count_include_pad_ ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
                                CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  CudnnPoolingDescriptor pooling_desc;
  LOTUS_RETURN_IF_ERROR(pooling_desc.Set(mode, kernel_shape, pads, strides));

  CUDNN_RETURN_IF_ERROR(cudnnPoolingForward(CudnnHandle(), pooling_desc, &alpha, x_tensor, x_data, &beta, y_tensor, y_data));

  return Status::OK();
}

}  // namespace Cuda
}  // namespace Lotus
