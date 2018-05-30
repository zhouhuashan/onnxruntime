#include "core/providers/cpu/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Tile")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Tile<float>);

template <>
Status Tile<float>::Compute(OpKernelContext *ctx) const {
  auto &input_tensor = *ctx->Input<Tensor>(0);
  auto &repeats_tensor = *ctx->Input<Tensor>(1);
  size_t dimension_count = input_tensor.Shape().NumDimensions();

  if (repeats_tensor.Shape().NumDimensions() != 1)
    return Status(LOTUS, INVALID_ARGUMENT, "'repeat' input tensor must be 1 dimensional");
  if (size_t(repeats_tensor.Shape().Size()) != input_tensor.Shape().NumDimensions())
    return Status(LOTUS, INVALID_ARGUMENT, "'repeat' input tensor must have the same length as the 'input' tensor");

  // Calculate the shape of the output tensor
  auto *repeats = repeats_tensor.Data<int64_t>();
  std::vector<int64_t> output_dims = input_tensor.Shape().GetDims();
  for (auto axis = 0; axis < input_tensor.Shape().NumDimensions(); axis++)
    output_dims[axis] *= repeats[axis];
  TensorShape outputShape(output_dims);
  auto &output_tensor = *ctx->Output(0, outputShape);

  auto *output = output_tensor.MutableData<float>();
  auto *input = input_tensor.Data<float>();

  TensorPitches output_pitches(output_tensor);
  TensorAxisCounters input_counters(input_tensor);

  while (input_counters) {
    // Copy the input data over
    size_t input_pitch = input_tensor.Shape().GetDims().back();
    for (size_t i = 0; i < input_pitch; i++)
      *output++ = *input++;

    // Tile it for the innermost axis
    const auto *copy = output - input_tensor.Shape()[dimension_count - 1];
    for (int64_t repeat = (repeats[dimension_count - 1] - 1) * input_pitch; repeat-- > 0;)
      *output++ = *copy++;

    // Tile it in the other axes
    while (input_counters.Increment()) {
      ptrdiff_t pitch = output_pitches[input_counters.Axis()] * input_tensor.Shape()[input_counters.Axis()];
      copy = output - pitch;
      for (int64_t repeat = (repeats[input_counters.Axis()] - 1) * pitch; repeat-- > 0;) {
        *output++ = *copy++;
      }
    }
  }
  return Status::OK();
}
}  // namespace Lotus
