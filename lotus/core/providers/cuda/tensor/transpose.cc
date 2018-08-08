#include "transpose.h"
#include "transpose_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Transpose,                                                  \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Transpose<T>);

template <typename T>
Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& X = *ctx->Input<Tensor>(0);
  const TensorShape& input_shape = X.Shape();
  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  std::vector<int64_t> output_dims(rank);
  std::vector<int64_t> default_perm(rank);
  const std::vector<int64_t>* p_perm = nullptr;
  ComputeOutputShape(X, output_dims, default_perm, p_perm);

  TensorShape output_shape{output_dims};
  Tensor* Y = ctx->Output(0, output_shape);

  ResetScratchBuffer();
  CudaAsyncBuffer<int64_t> input_strides(this, rank);
  CudaAsyncBuffer<int64_t> perm(this, *p_perm);
  CudaAsyncBuffer<fast_divmod> fdm_output_strides(this, rank);
  LOTUS_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_dims));
  LOTUS_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));

  LOTUS_RETURN_IF_ERROR(input_strides.CopyToGpu());
  LOTUS_RETURN_IF_ERROR(perm.CopyToGpu());
  LOTUS_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());

  TransposeImpl(
      rank,
      input_strides.GpuPtr(),
      perm.GpuPtr(),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X.Data<T>()),
      fdm_output_strides.GpuPtr(),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y->MutableData<T>()),
      output_shape.Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace Cuda
}  // namespace Lotus
