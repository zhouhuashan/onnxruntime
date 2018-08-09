#include "core/providers/cpu/tensor/concat.h"

namespace Lotus {

ONNX_CPU_OPERATOR_KERNEL(
    Concat,
    4,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

Status ConcatBase::PrepareForCompute(OpKernelContext* ctx, int input_count, Prepare& p) const {
  LOTUS_RETURN_IF_NOT(input_count >= 1, "Must have 1 or more inputs");

  auto& inputs_0 = *ctx->Input<Tensor>(0);

  // Ensure all of the non concatenated axes match each other
  for (int index = 1; index < input_count; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    // Ensure all the other axes match
    auto dimension_count = inputs_0.Shape().NumDimensions();
    for (int axis_index = 0; axis_index < dimension_count; axis_index++) {
      if (axis_index == axis_)
        continue;
      LOTUS_RETURN_IF_NOT(data_n.Shape()[axis_index] == inputs_0.Shape()[axis_index], "Non concat axis dimensions must match: Axis ", axis_index, " has mismatched dimensions of ", data_n.Shape()[axis_index], " and ", inputs_0.Shape()[axis_index]);
    }
  }

  // Calculate the size of the concatenated axis, and verify all other dimensions match
  size_t concat_axis_size = 0;
  for (int index = 0; index < input_count; index++) {
    concat_axis_size += ctx->Input<Tensor>(index)->Shape()[int(axis_)];
  }

  // Calculate the shape of the output tensor
  std::vector<int64_t> dims;
  for (int dimension_index = 0; dimension_index < inputs_0.Shape().NumDimensions(); dimension_index++)
    dims.emplace_back(inputs_0.Shape()[dimension_index]);
  dims[axis_] = concat_axis_size;
  TensorShape outputShape(dims);

  // The output_axis_pitch is the number of elements to add to move to the next split axis in the output
  p.output_axis_pitch = 1;
  for (auto i = int64_t(dims.size()); i-- > axis_;)
    p.output_axis_pitch *= dims[i];

  auto& concat_result = *ctx->Output(0, outputShape);
  p.output_tensor = &concat_result;

  for (int input_index = 0; input_index < input_count; input_index++) {
    auto& data_n = *ctx->Input<Tensor>(input_index);

    LOTUS_RETURN_IF_NOT(data_n.DataType() == concat_result.DataType());

    // The input_axis_pitch is the number of elements to add to move to the next split axis in the input
    int64_t input_axis_pitch = 1;
    for (int i = int(data_n.Shape().NumDimensions()); i-- > axis_;)
      input_axis_pitch *= data_n.Shape()[i];

    p.inputs.push_back({&data_n, input_axis_pitch});
  }

  return Status::OK();
}

Status Concat::Compute(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();

  Prepare p;
  LOTUS_RETURN_IF_ERROR(PrepareForCompute(ctx, input_count, p));

  auto is_string_type = ctx->Input<Tensor>(0)->DataType() == DataTypeImpl::GetType<std::string>();

  int64_t output_offset = 0;
  auto element_bytes = p.output_tensor->DataType()->Size();
  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];
    auto input_axis_pitch = prep.axis_pitch;
    const uint8_t* input = static_cast<const uint8_t*>(prep.tensor->DataRaw());
    auto input_size = prep.tensor->Shape().Size();

    // Copy the data across. For every 'input_axis_pitch' values copied, we move over by the 'output_axis_pitch'
    uint8_t* output = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());
    for (int idxCopy = 0; idxCopy < input_size / input_axis_pitch; ++idxCopy) {
      if (is_string_type) {
        for (int idxItem = 0; idxItem < input_axis_pitch; ++idxItem)
          reinterpret_cast<std::string*>(output)[output_offset + idxCopy * p.output_axis_pitch + idxItem] =
              reinterpret_cast<const std::string*>(input)[idxCopy * input_axis_pitch + idxItem];
      } else
        memcpy(
            output + (output_offset + idxCopy * p.output_axis_pitch) * element_bytes,
            input + idxCopy * input_axis_pitch * element_bytes,
            input_axis_pitch * element_bytes);
    }
    output_offset += input_axis_pitch;
  }
  return Status::OK();
}

}  // namespace Lotus
