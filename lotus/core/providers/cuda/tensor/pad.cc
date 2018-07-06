#include "pad.h"
#include "pad_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                              \
  REGISTER_KERNEL(KernelDefBuilder("Pad")                                     \
                      .Domain(LotusIR::kOnnxDomain)                           \
                      .SinceVersion(2)                                        \
                      .Provider(LotusIR::kCudaExecutionProvider)              \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                  Pad<T>);

template <typename T>
Status Pad<T>::Compute(OpKernelContext *ctx) const {
  const auto &input_tensor = *ctx->Input<Tensor>(0);
  const auto &input_dims = input_tensor.Shape().GetDims();
  TensorPitches input_pitches(input_tensor);
  std::vector<int64_t> output_dims(input_dims);
  size_t dimension_count = output_dims.size();

  LOTUS_ENFORCE(dimension_count * 2 == pads_.size(), "'pads' attribute has wrong number of values");

  // Calculate output dimensions, and handle any negative padding
  std::vector<int64_t> lower_pads(dimension_count);
  std::vector<int64_t> upper_pads(dimension_count);
  for (size_t i = 0; i < dimension_count; i++) {
    lower_pads[i] = pads_[i] + slices_[i];
    upper_pads[i] = pads_[i + dimension_count] + slices_[i + dimension_count];
    output_dims[i] += lower_pads[i] + upper_pads[i];
  }
  TensorShape output_shape(output_dims);
  FastDivModStrides fdm_output_strides(output_dims);
  auto &output_tensor = *ctx->Output(0, output_shape);

  IAllocatorUniquePtr<int64_t> input_dims_cuda;
  IAllocatorUniquePtr<int64_t> input_strides_cuda;
  IAllocatorUniquePtr<int64_t> lower_pads_cuda;
  IAllocatorUniquePtr<int64_t> upper_pads_cuda;
  IAllocatorUniquePtr<fast_divmod> fdm_output_strides_cuda;
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(input_dims_cuda, input_dims));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(input_strides_cuda, input_pitches));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(lower_pads_cuda, lower_pads));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(upper_pads_cuda, upper_pads));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(fdm_output_strides_cuda, fdm_output_strides.GetStrides()));

  PadImpl(
      dimension_count,
      input_dims_cuda.get(),
      input_strides_cuda.get(),
      lower_pads_cuda.get(),
      upper_pads_cuda.get(),
      value_,
      static_cast<int>(mode_),
      reinterpret_cast<const typename ToCudaType<T>::MappedType *>(input_tensor.Data<T>()),
      fdm_output_strides_cuda.get(),
      reinterpret_cast<typename ToCudaType<T>::MappedType *>(output_tensor.MutableData<T>()),
      output_tensor.Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Pad<T>::Compute(OpKernelContext *ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace Cuda
};  // namespace Lotus
