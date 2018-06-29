#include "core/providers/cuda/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"
#include "tile_impl.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                              \
  REGISTER_KERNEL(KernelDefBuilder("Tile")                                    \
                      .Domain(LotusIR::kOnnxDomain)                           \
                      .SinceVersion(1)                                        \
                      .Provider(LotusIR::kCudaExecutionProvider)              \
                      .InputMemoryType<kMemTypeCPUInput>(1)                   \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                  Tile<T>);

template <typename T>
Status Tile<T>::Compute(OpKernelContext *ctx) const {
  auto &input_tensor = *ctx->Input<Tensor>(0);
  auto &repeats_tensor = *ctx->Input<Tensor>(1);
  size_t rank = input_tensor.Shape().NumDimensions();

  if (repeats_tensor.Shape().NumDimensions() != 1)
    return Status(LOTUS, INVALID_ARGUMENT, "'repeat' input tensor must be 1 dimensional");
  if (size_t(repeats_tensor.Shape().Size()) != rank)
    return Status(LOTUS, INVALID_ARGUMENT, "'repeat' input tensor must have the same length as the 'input' tensor");

  // Calculate the shape of the output tensor
  auto *repeats = repeats_tensor.Data<int64_t>();
  const auto &input_shape = input_tensor.Shape().GetDims();
  std::vector<int64_t> output_dims(input_shape);
  for (auto axis = 0; axis < rank; axis++)
    output_dims[axis] *= repeats[axis];
  TensorShape outputShape(output_dims);
  auto &output_tensor = *ctx->Output(0, outputShape);

  T *output_data = output_tensor.MutableData<T>();
  const T *input_data = input_tensor.Data<T>();
  TensorPitches input_pitches(input_tensor);
  FastDivModStrides fdm_output_strides(output_dims);

  std::vector<fast_divmod> fdm_input_shape;
  for (auto input_dim : input_shape)
    fdm_input_shape.emplace_back(fast_divmod(gsl::narrow_cast<int>(input_dim)));

  // allocate temp memory for offset arrays
  IAllocatorUniquePtr<int64_t> input_stride_cuda;
  IAllocatorUniquePtr<fast_divmod> fdm_input_shape_cuda, fdm_output_strides_cuda;
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(fdm_input_shape_cuda, fdm_input_shape));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(input_stride_cuda, input_pitches));
  LOTUS_RETURN_IF_ERROR(CopySmallVectorToGPU(fdm_output_strides_cuda, fdm_output_strides.GetStrides()));

  TileImpl(
      rank,
      fdm_input_shape_cuda.get(),
      input_stride_cuda.get(),
      reinterpret_cast<const typename ToCudaType<T>::MappedType *>(input_data),
      fdm_output_strides_cuda.get(),
      reinterpret_cast<typename ToCudaType<T>::MappedType *>(output_data),
      output_tensor.Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Tile<T>::Compute(OpKernelContext *ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace Cuda
}  // namespace Lotus
