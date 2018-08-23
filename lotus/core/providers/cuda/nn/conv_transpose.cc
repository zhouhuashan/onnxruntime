#include "conv_transpose.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_OP_TYPED(T)                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      ConvTranspose,                                                            \
      kOnnxDomain,                                                              \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T>);

REGISTER_OP_TYPED(float)
REGISTER_OP_TYPED(double)
REGISTER_OP_TYPED(MLFloat16)

template <typename T>
Status ConvTranspose<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  bool has_bias = (num_inputs == 3);
  bool input_dims_changed = (s_.last_x_dims != x_dims);
  bool w_dims_changed = (s_.last_w_dims != w_dims);

  CudaT* y_data = nullptr;
  if (input_dims_changed || w_dims_changed) {
    if (input_dims_changed)
      s_.last_x_dims = x_dims;

    if (w_dims_changed)
      s_.last_w_dims = w_dims;

    Prepare p;
    LOTUS_RETURN_IF_ERROR(PrepareForCompute(context, has_bias, p));

    const auto& y_dims = p.Y->Shape().GetDims();
    s_.y_dims = y_dims;

    LOTUS_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims, CudnnTensor::GetDataType<CudaT>()));
    LOTUS_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims, CudnnTensor::GetDataType<CudaT>()));

    if (w_dims_changed)
      LOTUS_RETURN_IF_ERROR(s_.filter_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    LOTUS_RETURN_IF_ERROR(s_.conv_desc.Set(p.kernel_shape.size(), p.pads, p.strides, p.dilations, mode, CudnnTensor::GetDataType<CudaT>()));
    CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(s_.conv_desc, gsl::narrow_cast<int>(group_)));

    s_.found_algo = false;

    if (has_bias) {
      const auto& b_shape = p.B->Shape();
      LOTUS_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
      std::vector<int64_t> b_dims(2 + p.kernel_shape.size());
      b_dims[0] = 1;           // N
      b_dims[1] = b_shape[0];  // C
      for (int i = 0; i < p.kernel_shape.size(); i++)
        b_dims[2 + i] = 1;

      LOTUS_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
    }

    y_data = reinterpret_cast<CudaT*>(p.Y->MutableData<T>());
  } else {
    Tensor* Y = context->Output(0, s_.y_dims);
    y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  }
  auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto w_data = reinterpret_cast<const CudaT*>(W->Data<T>());

  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes);

  if (!s_.found_algo) {
    cudnnConvolutionBwdDataAlgoPerf_t perf;
    int algo_count = 1;
    CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
        CudnnHandle(),
        s_.filter_desc,
        w_data,
        s_.x_tensor,
        x_data,
        s_.conv_desc,
        s_.y_tensor,
        y_data,
        1,
        &algo_count,
        &perf,
        workspace.get(),
        s_.workspace_bytes));
    CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
    s_.algo = perf.algo;
    s_.workspace_bytes = perf.memory;
    workspace = GetScratchBuffer<void>(s_.workspace_bytes);
    s_.found_algo = true;
  }

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardData(
          CudnnHandle(),
          &alpha,
          s_.filter_desc,
          w_data,
          s_.x_tensor,
          x_data,
          s_.conv_desc,
          s_.algo,
          workspace.get(),
          s_.workspace_bytes,
          &beta,
          s_.y_tensor,
          y_data));

  if (has_bias) {
    const Tensor* B = context->Input<Tensor>(2);
    auto b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, s_.b_tensor, b_data, &alpha, s_.y_tensor, y_data));
  }

  return Status::OK();
}

}  // namespace Cuda
}  // namespace Lotus
