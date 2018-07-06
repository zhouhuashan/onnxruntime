#include "transpose.h"
#include "transpose_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                              \
  REGISTER_KERNEL(KernelDefBuilder("Transpose")                               \
                      .Domain(LotusIR::kOnnxDomain)                           \
                      .SinceVersion(1)                                        \
                      .Provider(LotusIR::kCudaExecutionProvider)              \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                  Transpose<T>);

template <typename T>
Status Transpose<T>::Compute(OpKernelContext* ctx) const {
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
  TensorPitches input_pitches(X);
  FastDivModStrides fdm_output_strides(output_dims);

  IAllocatorUniquePtr<int64_t> input_strides_cuda;
  IAllocatorUniquePtr<int64_t> perm_cuda;
  IAllocatorUniquePtr<fast_divmod> fdm_output_strides_cuda;
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(input_strides_cuda, input_pitches));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(perm_cuda, *p_perm));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(fdm_output_strides_cuda, fdm_output_strides.GetStrides()));

  TransposeImpl(
      rank,
      input_strides_cuda.get(),
      perm_cuda.get(),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X.Data<T>()),
      fdm_output_strides_cuda.get(),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y->MutableData<T>()),
      output_shape.Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Transpose<T>::Compute(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace Cuda
}  // namespace Lotus
