#include "core/providers/cuda/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"
#include "tile_impl.h"

namespace Lotus {
namespace Cuda {
REGISTER_KERNEL(KernelDefBuilder("Tile")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .InputMemoryType<kMemTypeCPUInput>(1)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Tile<float>);

template <>
Status Tile<float>::Compute(OpKernelContext *ctx) const {
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

  auto *output = output_tensor.MutableData<float>();
  auto *input = input_tensor.Data<float>();
  TensorPitches input_pitches(input_tensor);
  TensorPitches output_pitches(output_tensor);

  // allocate temp memory for offset arrays
  IAllocatorUniquePtr<int64_t> input_shape_cuda, input_stride_cuda, output_stride_cuda;
  CopySmallVectorToGPU(input_shape_cuda, input_shape);
  CopySmallVectorToGPU(input_stride_cuda, input_pitches);
  CopySmallVectorToGPU(output_stride_cuda, output_pitches);

  TileImpl(
      rank,
      input_shape_cuda.get(),
      input_stride_cuda.get(),
      input,
      output_stride_cuda.get(),
      output,
      output_tensor.Shape().Size());

  return Status::OK();
}
}  // namespace Cuda
}  // namespace Lotus
