
#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/conv.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_OP_TYPED(T)                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Conv,                                                                     \
      kOnnxDomain,                                                              \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<float>);

REGISTER_OP_TYPED(float)
REGISTER_OP_TYPED(double)
REGISTER_OP_TYPED(MLFloat16)

template <typename T>
Status Conv<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* W = context->Input<Tensor>(1);
  // Get Bias tensor
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;
  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];

  LOTUS_RETURN_IF_ERROR(ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape = ComputeKernelShape(W->Shape());
  auto rank = kernel_shape.size();
  std::vector<int64_t> pads(pads_);
  if (pads.empty()) {
    pads.resize(rank * 2, 0);
  }
  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty()) {
    dilations.resize(rank, 1);
  }
  std::vector<int64_t> strides(strides_);
  if (strides.empty()) {
    strides.resize(rank, 1);
  }

  std::vector<int64_t> y_dims;
  y_dims.insert(y_dims.begin(), {N, M});
  InferOutputShape(x_shape.Slice(2), kernel_shape, strides, dilations, &pads, &y_dims);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  const TensorShape& w_shape = W->Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();
  std::vector<int64_t> x_dims_cudnn = x_dims;
  std::vector<int64_t> y_dims_cudnn = y_dims;
  if (rank < 2) {
    // cudnn only takes 4D or 5D input, so pad dimensions if needed
    x_dims_cudnn.push_back(1);
    y_dims_cudnn.push_back(1);
    w_dims.push_back(1);
    pads.insert(pads.begin() + rank, 0);
    pads.insert(pads.end(), 0);
    kernel_shape.push_back(1);
    strides.push_back(1);
    dilations.push_back(1);
  }

  auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto w_data = reinterpret_cast<const CudaT*>(W->Data<T>());
  auto y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor x_tensor;
  CudnnTensor y_tensor;
  LOTUS_RETURN_IF_ERROR(x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
  LOTUS_RETURN_IF_ERROR(y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));

  CudnnFilterDescriptor w_tensor;
  LOTUS_RETURN_IF_ERROR(w_tensor.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

  CudnnConvolutionDescriptor conv_desc;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  LOTUS_RETURN_IF_ERROR(conv_desc.Set(kernel_shape.size(), pads, strides, dilations, mode, CudnnTensor::GetDataType<CudaT>()));
  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(conv_desc, gsl::narrow_cast<int>(group_)));

  cudnnConvolutionFwdAlgo_t algo;
  auto algo_cache_it = algo_cache_.find(x_dims);
  if (algo_cache_it != algo_cache_.end()) {
    algo = algo_cache_it->second;
  } else {
    cudnnConvolutionFwdAlgoPerf_t perf;
    int algo_count = 1;
    CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(CudnnHandle(),
                                                                 x_tensor,
                                                                 w_tensor,
                                                                 conv_desc,
                                                                 y_tensor,
                                                                 1,
                                                                 &algo_count,
                                                                 &perf));
    algo = perf.algo;
    algo_cache_.insert(std::make_pair(x_dims, algo));
  }

  size_t workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(CudnnHandle(),
                                                                x_tensor,
                                                                w_tensor,
                                                                conv_desc,
                                                                y_tensor,
                                                                algo,
                                                                &workspace_bytes));

  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(workspace_bytes);

  CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(CudnnHandle(),
                                                &alpha,
                                                x_tensor,
                                                x_data,
                                                w_tensor,
                                                w_data,
                                                conv_desc,
                                                algo,
                                                workspace.get(),
                                                workspace_bytes,
                                                &beta,
                                                y_tensor,
                                                y_data));

  if (B) {
    const auto& b_shape = B->Shape();
    LOTUS_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
    std::vector<int64_t> b_dims(2 + kernel_shape.size());
    b_dims[0] = 1;           // N
    b_dims[1] = b_shape[0];  // C
    for (int i = 0; i < kernel_shape.size(); i++)
      b_dims[2 + i] = 1;

    CudnnTensor b_tensor;
    LOTUS_RETURN_IF_ERROR(b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));

    auto b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, b_tensor, b_data, &alpha, y_tensor, y_data));
  }

  return Status::OK();
}

CudnnFilterDescriptor::CudnnFilterDescriptor() : desc_(nullptr) {
}

CudnnFilterDescriptor::~CudnnFilterDescriptor() {
  if (desc_ != nullptr) {
    cudnnDestroyFilterDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status CudnnFilterDescriptor::Set(const std::vector<int64_t>& filter_dims, cudnnDataType_t data_type) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateFilterDescriptor(&desc_));

  int rank = gsl::narrow_cast<int>(filter_dims.size());
  std::vector<int> w_dims(rank);
  for (int i = 0; i < rank; i++) {
    w_dims[i] = gsl::narrow_cast<int>(filter_dims[i]);
  }

  CUDNN_RETURN_IF_ERROR(cudnnSetFilterNdDescriptor(desc_,
                                                   data_type,
                                                   CUDNN_TENSOR_NCHW,
                                                   rank,
                                                   w_dims.data()));
  return Status::OK();
}

CudnnConvolutionDescriptor::CudnnConvolutionDescriptor() : desc_(nullptr) {
}

CudnnConvolutionDescriptor::~CudnnConvolutionDescriptor() {
  if (desc_ != nullptr) {
    cudnnDestroyConvolutionDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status CudnnConvolutionDescriptor::Set(
    size_t rank,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dilations,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t data_type) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateConvolutionDescriptor(&desc_));

  std::vector<int> pad_dims(rank);
  std::vector<int> stride_dims(rank);
  std::vector<int> dilation_dims(rank);
  for (int i = 0; i < rank; i++) {
    pad_dims[i] = gsl::narrow_cast<int>(pads[i]);
    stride_dims[i] = gsl::narrow_cast<int>(strides[i]);
    dilation_dims[i] = gsl::narrow_cast<int>(dilations[i]);
  }

  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionNdDescriptor(
      desc_,
      gsl::narrow_cast<int>(rank),
      pad_dims.data(),
      stride_dims.data(),
      dilation_dims.data(),
      mode,
      data_type));

  return Status::OK();
}

}  // namespace Cuda
}  // namespace Lotus
