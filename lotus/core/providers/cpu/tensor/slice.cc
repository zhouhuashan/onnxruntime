#include "core/providers/cpu/tensor/slice.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Slice")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Slice<float>);

// std::clamp doesn't exist until C++17 so create a local version
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

template <>
Status Slice<float>::Compute(OpKernelContext* ctx) const {
  auto& input_tensor = *ctx->Input<Tensor>(0);
  auto& input_dimensions = input_tensor.Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> output_dims;
  for (size_t i = 0; i < dimension_count; i++)
    output_dims.push_back(input_dimensions[i]);

  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<size_t> axes;
  if (axes_.size()) {
    for (auto axis : axes_)
      axes.push_back(axis);
  } else {
    for (size_t i = 0; i < starts.size(); i++)
      axes.push_back(i);
  }

  if (axes.size() > starts_.size())
    return Status(LOTUS, INVALID_ARGUMENT, "'axes' has more entries than the 'starts' attribute holds");
  if (axes.size() > ends_.size())
    return Status(LOTUS, INVALID_ARGUMENT, "'axes' has more entries than the 'ends' attribute holds");

  // Iterate through the provided axes and override the start/end ranges
  for (size_t axesIndex = 0; axesIndex < axes.size(); axesIndex++) {
    auto axis = axes[axesIndex];
    if (axis >= dimension_count)
      return Status(LOTUS, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    auto start = starts_[axesIndex];
    if (start < 0)
      start += input_dimensions[axis];
    starts[axis] = clamp(start, int64_t{0}, input_dimensions[axis]);

    auto end = ends_[axesIndex];
    if (end < 0)
      end += input_dimensions[axis];
    output_dims[axis] = clamp(end, int64_t{0}, input_dimensions[axis]) - starts[axis];
    if (output_dims[axis] <= 0)
      return Status(LOTUS, INVALID_ARGUMENT, "'starts' and 'ends' values resulted in a negative dimension");
  }

  // Calculate input pitches
  std::vector<int64_t> input_pitches(dimension_count, 0);
  input_pitches.back() = 1;
  for (size_t i = dimension_count - 1; i-- > 0;) {
    input_pitches[i] = input_pitches[i + 1] * input_tensor.Shape().GetDims()[i + 1];
  }

  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.MutableData<float>();
  auto* output_end = output + output_shape.Size();
  auto* input = input_tensor.Data<float>();

  // Initial skip, so that input points to the first element to copy
  for (size_t i = 0; i < dimension_count; i++)
    input += input_pitches[i] * starts[i];

  std::vector<int64_t> axis_index(dimension_count - 1, 0);
  for (;;) {
    size_t axis = dimension_count - 1;  // Start in the innermost axis

    // Special case the innermost axis loop
    size_t size = output_dims[axis];
    for (size_t i = 0; i < size; i++)
      *output++ = input[i];
    if (output == output_end)  // For a 1D tensor there is no next outer axis, so check the end condition here
      break;
    input += input_pitches[axis - 1];  // Move once in the next outer axis

    while (++axis_index[--axis] == output_dims[axis]) {
      axis_index[axis] = 0;
      input += input_pitches[axis - 1] - output_dims[axis] * input_pitches[axis];  // Reset to start of this axis, and move once in the next outer axis
    }
  }

  return Status::OK();
}

}  // namespace Lotus
