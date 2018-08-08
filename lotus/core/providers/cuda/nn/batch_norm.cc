#include "batch_norm.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
using namespace std;
namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      BatchNormalization,                                            \
      kOnnxDomain,                                                   \
      7,                                                             \
      T,                                                             \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("X", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("scale", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("mean", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("var", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);

struct BNTensorDescriptor final {
  ~BNTensorDescriptor() {
    if (derived_bn_desc) {
      cudnnDestroyTensorDescriptor(derived_bn_desc);
      derived_bn_desc = nullptr;
    }
  }

  Status Set(const CudnnTensor& x_desc, cudnnBatchNormMode_t mode) {
    if (!derived_bn_desc) {
      CUDNN_RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&derived_bn_desc));
    }
    CUDNN_RETURN_IF_ERROR(cudnnDeriveBNTensorDescriptor(derived_bn_desc, x_desc, mode));
    return Status::OK();
  }

  cudnnTensorDescriptor_t derived_bn_desc = nullptr;
};

static void NormalizeDims(const TensorShape& x_shape, std::vector<int64_t>& new_dims) {
  new_dims.clear();
  auto& orig_dims = x_shape.GetDims();
  if (orig_dims.size() == 4 /*supported size by CUDA*/ ||
      orig_dims.size() == 5 /*supported size by CUDA*/) {
    new_dims = orig_dims;
    return;
  }

  auto rank = x_shape.NumDimensions();
  auto num_samples = rank > 0 ? orig_dims[0] : 1;  // NCHW
  auto num_channels = rank > 1 ? orig_dims[1] : 1;
  auto width = rank > 3 ? orig_dims[3] : 1;
  auto height = rank > 2 ? orig_dims[2] : 1;
  new_dims = {num_samples, num_channels, height, width};
}

template <typename T>
Status BatchNorm<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* var = p_op_kernel_context->Input<Tensor>(4);

  LOTUS_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var));

  const TensorShape& x_shape = X->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  auto y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->Data<T>());
  auto b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
  auto mean_data = reinterpret_cast<const CudaT*>(mean->Data<T>());
  auto var_data = reinterpret_cast<const CudaT*>(var->Data<T>());

  CudnnTensor data_desc;
  vector<int64_t> new_dims;
  NormalizeDims(x_shape, new_dims);
  LOTUS_RETURN_IF_ERROR(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));

  BNTensorDescriptor bn_tensor_desc;
  LOTUS_RETURN_IF_ERROR(bn_tensor_desc.Set(data_desc, cudnn_batch_norm_mode_));

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardInference(
      CudnnHandle(),
      cudnn_batch_norm_mode_,
      &alpha,
      &beta,
      data_desc,
      x_data,
      data_desc,
      y_data,
      bn_tensor_desc.derived_bn_desc,
      scale_data,
      b_data,
      mean_data,
      var_data,
      epsilon_));

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status BatchNorm<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace Cuda
}  // namespace Lotus
