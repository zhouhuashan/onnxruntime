#include "core/providers/cpu/tensor/tile.h"

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
  auto &tiles_tensor = *ctx->Input<Tensor>(1);
  if (tiles_tensor.Shape().NumDimensions() != 0)
    return Status(LOTUS, INVALID_ARGUMENT, "'tiles' tensor must be a scalar");
  auto tiles = *tiles_tensor.Data<int64_t>();
  if (tiles <= 0)
    return Status(LOTUS, INVALID_ARGUMENT, "'tiles' value must be greater than zero");
  auto &axis_tensor = *ctx->Input<Tensor>(2);
  if (axis_tensor.Shape().NumDimensions() != 0)
    return Status(LOTUS, INVALID_ARGUMENT, "'axis' tensor must be a scalar");
  size_t axis = *axis_tensor.Data<int64_t>();
  if (axis >= input_tensor.Shape().NumDimensions())
    return Status(LOTUS, INVALID_ARGUMENT, "'axis' must be within the dimensions of the input tensor");

  // Calculate the shape of the output tensor
  std::vector<int64_t> dims = input_tensor.Shape().GetDims();
  dims[axis] *= tiles;
  TensorShape outputShape(dims);
  auto &output_tensor = *ctx->Output(0, outputShape);

  int64_t tilePitch = 1;
  for (auto i = input_tensor.Shape().NumDimensions(); i-- > axis;)
    tilePitch *= input_tensor.Shape()[i];

  auto *output = output_tensor.MutableData<float>();
  auto *output_end = output + output_tensor.Shape().Size();
  auto *input = input_tensor.Data<float>();

  // Tiling is done by copying 'tilePitch' number of entries 'tiles' times, then repeating for the remaining axes
  while (output != output_end) {
    for (int tile_index = 0; tile_index < tiles; tile_index++) {
      for (int j = 0; j < tilePitch; j++)
        *output++ = input[j];
    }
    input += tilePitch;
  }
  return Status::OK();
}
}  // namespace Lotus
