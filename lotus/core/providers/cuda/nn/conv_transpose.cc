#include "conv_transpose.h"
#include "conv.h"

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
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  Prepare p;
  LOTUS_RETURN_IF_ERROR(PrepareForCompute(context, num_inputs == 3, p));

  auto x_data = reinterpret_cast<const CudaT*>(p.X->Data<T>());
  auto y_data = reinterpret_cast<CudaT*>(p.Y->MutableData<T>());
  const auto& x_dims = p.X->Shape().GetDims();
  const auto& y_dims = p.Y->Shape().GetDims();
  std::vector<int64_t> w_dims = p.F->Shape().GetDims();
  auto w_data = reinterpret_cast<const CudaT*>(p.F->Data<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor x_tensor;
  CudnnTensor y_tensor;
  LOTUS_RETURN_IF_ERROR(x_tensor.Set(x_dims, CudnnTensor::GetDataType<CudaT>()));
  LOTUS_RETURN_IF_ERROR(y_tensor.Set(y_dims, CudnnTensor::GetDataType<CudaT>()));

  CudnnFilterDescriptor w_tensor;
  LOTUS_RETURN_IF_ERROR(w_tensor.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

  CudnnConvolutionDescriptor conv_desc;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  LOTUS_RETURN_IF_ERROR(conv_desc.Set(p.kernel_shape.size(), p.pads, p.strides, p.dilations, mode, CudnnTensor::GetDataType<CudaT>()));
  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(conv_desc, gsl::narrow_cast<int>(group_)));

  cudnnConvolutionBwdDataAlgo_t algo;
  auto algo_cache_it = algo_cache_.find(x_dims);
  if (algo_cache_it != algo_cache_.end()) {
    algo = algo_cache_it->second;
  } else {
    cudnnConvolutionBwdDataAlgoPerf_t perf;
    int algo_count = 1;
    CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        CudnnHandle(),
        w_tensor,
        x_tensor,
        conv_desc,
        y_tensor,
        1,
        &algo_count,
        &perf));
    algo = perf.algo;
    algo_cache_.insert(std::make_pair(x_dims, algo));
  }

  size_t workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
      CudnnHandle(),
      w_tensor,
      x_tensor,
      conv_desc,
      y_tensor,
      algo,
      &workspace_bytes));

  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(workspace_bytes);

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardData(
          CudnnHandle(),
          &alpha,
          w_tensor,
          w_data,
          x_tensor,
          x_data,
          conv_desc,
          algo,
          workspace.get(),
          workspace_bytes,
          &beta,
          y_tensor,
          y_data));

  if (p.B) {
    const auto& b_shape = p.B->Shape();
    LOTUS_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
    std::vector<int64_t> b_dims(2 + p.kernel_shape.size());
    b_dims[0] = 1;           // N
    b_dims[1] = b_shape[0];  // C
    for (int i = 0; i < p.kernel_shape.size(); i++)
      b_dims[2 + i] = 1;

    CudnnTensor b_tensor;
    LOTUS_RETURN_IF_ERROR(b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));

    auto b_data = reinterpret_cast<const CudaT*>(p.B->Data<T>());
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, b_tensor, b_data, &alpha, y_tensor, y_data));
  }

  return Status::OK();
}

}  // namespace Cuda
}  // namespace Lotus
