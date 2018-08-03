#include "core/providers/cpu/tensor/concat.h"

namespace Lotus {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Concat,
    4,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Concat,
    4,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Concat<int32_t>);

template <typename T>
Status Concat<T>::Compute(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();
  LOTUS_ENFORCE(input_count >= 1, "Must have 1 or more inputs");

  auto& inputs_0 = *ctx->Input<Tensor>(0);

  // Ensure all of the non concatenated axes match each other
  for (int index = 1; index < input_count; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    // Ensure all the other axes match
    auto dimension_count = inputs_0.Shape().NumDimensions();
    for (int axis_index = 0; axis_index < dimension_count; axis_index++) {
      if (axis_index == axis_)
        continue;
      LOTUS_ENFORCE(data_n.Shape()[axis_index] == inputs_0.Shape()[axis_index], "Non concat axis dimensions must match: Axis ", axis_index, " has mismatched dimensions of ", data_n.Shape()[axis_index], " and ", inputs_0.Shape()[axis_index]);
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
  int64_t output_axis_pitch = 1;
  for (auto i = int64_t(dims.size()); i-- > axis_;)
    output_axis_pitch *= dims[i];

  auto& concat_result = *ctx->Output(0, outputShape);
  T* output_base = concat_result.MutableData<T>();

  for (int input_index = 0; input_index < input_count; input_index++) {
    auto& data_n = *ctx->Input<Tensor>(input_index);

    // The input_axis_pitch is the number of elements to add to move to the next split axis in the input
    int64_t input_axis_pitch = 1;
    for (int i = int(data_n.Shape().NumDimensions()); i-- > axis_;)
      input_axis_pitch *= data_n.Shape()[i];

    const T* input = data_n.Data<T>();
    auto input_size = data_n.Shape().Size();

    // Copy the data across. For every 'input_axis_pitch' values copied, we move over by the 'output_axis_pitch'
    T* output = output_base;
    for (int i = 0, j = 0; i < input_size; i++) {
      output[i] = input[i];
      if (++j == input_axis_pitch) {
        output += output_axis_pitch - input_axis_pitch;  // Subtract input_axis_pitch because output is being indexed by 'i'
        j = 0;
      }
    }
    output_base += input_axis_pitch;
  }
  return Status::OK();
}

}  // namespace Lotus
