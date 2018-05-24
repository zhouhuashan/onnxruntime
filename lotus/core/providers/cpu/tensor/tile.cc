#include "core/providers/cpu/tensor/tile.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Tile")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Tile<float>);

struct TensorPitches : std::vector<int64_t> {
  TensorPitches(const Tensor &tensor)
      : std::vector<int64_t>(tensor.Shape().NumDimensions(), 0) {
    size_t dimension_count = tensor.Shape().NumDimensions();
    auto &dims = tensor.Shape().GetDims();

    // The pitches is the size of the next inner axis. Aka the amount to move by one of the next inner axis.
    // For a tensor with shape(2,3,4,5) the values would be: (3*4*5, 4*5, 5, 1)
    // Note that the outermost '2' is never used, as you never need to move by the entire size of the outermost axis

    back() = 1;  // The innermost axis is 1 (single values)
    for (size_t i = dimension_count - 1; i-- > 0;) {
      operator[](i) = operator[](i + 1) * dims[i + 1];
    }
  }
};

// This class is to iterate through the axes of an arbitrarily shaped tensor
// For example, a tensor with shape (2,3,4) will be iterated in this order:
// (0,0,x) (0,1,x) (0,2,x) (1,0,x) (1,1,x) (1,2,x)
// Note: The innermost axis is not iterated over since it's always special cased
struct TensorAxisCounters {
  TensorAxisCounters(const Tensor &tensor) : tensor_(tensor) {
    dimension_count_ = tensor_.Shape().NumDimensions();
    indices_.resize(dimension_count_ - 1, 0);
    axis_ = dimension_count_ - 1;
  }

  // Returns true if there was a carry to the next axis
  bool Increment() {
    if (axis_ == 0) {
      running_ = false;
      return false;
    }

    if (++indices_[--axis_] != tensor_.Shape()[axis_]) {
      axis_ = dimension_count_ - 1;
      return false;
    }

    indices_[axis_] = 0;  // Reset the counter for this axis
    return true;          // There was a carry
  }

  size_t Axis() const { return axis_; }
  operator bool() const { return running_; }

 private:
  const Tensor &tensor_;
  size_t dimension_count_;
  bool running_{true};
  size_t axis_;
  std::vector<int64_t> indices_;  // There is no index for innermost axis since it's a special case
};

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
