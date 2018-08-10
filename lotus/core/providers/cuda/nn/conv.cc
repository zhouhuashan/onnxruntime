
#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"

namespace Lotus {
namespace Cuda {

Status BinaryElementwiseBroadcastPrepare(
    const Tensor* lhs_tensor,
    const Tensor* rhs_tensor,
    Tensor* output_tensor,
    BinaryElementwisePreparation* p,
    const TensorShape* override_lhs_shape,
    const TensorShape* override_rhs_shape);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    float,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    double,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Conv<double>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Conv<MLFloat16>);

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
  std::vector<int64_t> pads(pads_);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(strides_);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> y_dims;
  y_dims.insert(y_dims.begin(), {N, M});
  InferOutputShape(x_shape.Slice(2), kernel_shape, strides, dilations, &pads, &y_dims);
  Tensor* Y = context->Output(0, TensorShape(y_dims));
  ResetScratchBuffer();

  const TensorShape& w_shape = W->Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();
  std::vector<int64_t> x_dims_cudnn = x_dims;
  std::vector<int64_t> y_dims_cudnn = y_dims;
  if (kernel_shape.size() < 2) {
    // cudnn only takes 4D or 5D input, so pad dimensions if needed
    x_dims_cudnn.push_back(1);
    y_dims_cudnn.push_back(1);
    w_dims.push_back(1);
    pads.insert(pads.begin() + kernel_shape.size(), 0);
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
  LOTUS_RETURN_IF_ERROR(conv_desc.Set(kernel_shape, pads, strides, dilations, mode, CudnnTensor::GetDataType<CudaT>()));
  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(conv_desc, gsl::narrow_cast<int>(group_)));

  cudnnConvolutionFwdAlgo_t algo;
  auto algo_cache_it = algo_cache_.find(x_dims);
  if (algo_cache_it != algo_cache_.end()) {
    algo = algo_cache_it->second;
  } else {
    cudnnConvolutionFwdPreference_t preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm(CudnnHandle(),
                                                              x_tensor,
                                                              w_tensor,
                                                              conv_desc,
                                                              y_tensor,
                                                              preference,
                                                              0,  // no memory limit
                                                              &algo));
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

  void* workspace = GetScratchBuffer<void>(workspace_bytes);

  CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(CudnnHandle(),
                                                &alpha,
                                                x_tensor,
                                                x_data,
                                                w_tensor,
                                                w_data,
                                                conv_desc,
                                                algo,
                                                workspace,
                                                workspace_bytes,
                                                &beta,
                                                y_tensor,
                                                y_data));

  if (B) {
    std::vector<int64_t> overrided_B_dims = B->Shape().GetDims();
    for (int i = 0; i < kernel_shape.size(); i++)
      overrided_B_dims.push_back(1);

    TensorShape overrided_B_shape(overrided_B_dims);

    BinaryElementwisePreparation prepare(this);
    LOTUS_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(Y, B, Y, &prepare, nullptr, &overrided_B_shape));
    Impl_Add<CudaT>(
        prepare.output_rank_or_simple_broadcast,
        prepare.lhs_padded_strides.GpuPtr(),
        reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()),
        prepare.rhs_padded_strides.GpuPtr(),
        reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()),
        prepare.fdm_output_strides.GpuPtr(),
        prepare.fdm_H,
        prepare.fdm_C,
        reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
        prepare.output_tensor->Shape().Size());
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

Status CudnnConvolutionDescriptor::Set(const std::vector<int64_t>& kernel_shape,
                                       const std::vector<int64_t>& pads,
                                       const std::vector<int64_t>& strides,
                                       const std::vector<int64_t>& dilations,
                                       cudnnConvolutionMode_t mode,
                                       cudnnDataType_t data_type) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateConvolutionDescriptor(&desc_));

  int rank = gsl::narrow_cast<int>(kernel_shape.size());
  std::vector<int> pad_dims(rank);
  std::vector<int> stride_dims(rank);
  std::vector<int> dilation_dims(rank);
  for (int i = 0; i < rank; i++) {
    pad_dims[i] = gsl::narrow_cast<int>(pads[i]);
    stride_dims[i] = gsl::narrow_cast<int>(strides[i]);
    dilation_dims[i] = gsl::narrow_cast<int>(dilations[i]);
  }

  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionNdDescriptor(desc_,
                                                        rank,
                                                        pad_dims.data(),
                                                        stride_dims.data(),
                                                        dilation_dims.data(),
                                                        mode,
                                                        data_type));

  return Status::OK();
}

}  // namespace Cuda
}  // namespace Lotus
