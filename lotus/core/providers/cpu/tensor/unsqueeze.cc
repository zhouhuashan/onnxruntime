#include "core/providers/cpu/tensor/unsqueeze.h"
using namespace Lotus::Common;

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Unsqueeze")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Unsqueeze<float>);

template <>
Status Unsqueeze<float>::Compute(OpKernelContext *ctx) const {
  auto &input_tensor = *ctx->Input<Tensor>(0);

  // New dimension count is the current dimensions + the number of entries in axes_
  // Initialize output_dims to 0 in each axis initially
  std::vector<int64_t> output_dims(axes_.size() + input_tensor.Shape().GetDims().size(), 0);

  // Set all axes_ indices to 1 in output_dims and check for duplicates
  for (size_t axis : axes_) {
    if (axis >= output_dims.size())
      return Status(LOTUS, INVALID_ARGUMENT, "'axes' has an out of range axis");
    if (output_dims[axis] != 0)
      return Status(LOTUS, INVALID_ARGUMENT, "'axes' has a duplicate axis");
    output_dims[axis] = 1;
  }

  // Now fill in the zero entries with the existing shape
  {
    auto begin = input_tensor.Shape().GetDims().cbegin();
    for (auto &axisSize : output_dims) {
      if (axisSize == 0)
        axisSize = *begin++;
    }
    assert(begin == input_tensor.Shape().GetDims().cend());
  }

  TensorShape output_shape(output_dims);
  auto &output_tensor = *ctx->Output(0, output_shape);
  auto *output = output_tensor.MutableData<float>();
  auto *input = input_tensor.Data<float>();

  // Copy the tensor
  size_t size = output_shape.Size();
  for (size_t i = 0; i < size; ++i)
    output[i] = input[i];
  // TODO(RyanHill): Optimize away copy when we can do an inplace operation

  return Status::OK();
}
}  // namespace Lotus
