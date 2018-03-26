#include "core/providers/cpu/misc/concat.h"

namespace Lotus {

REGISTER_KERNEL(KernelDef("Concat")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Concat<float>);

template <>
Status Concat<float>::compute(OpKernelContext* ctx) const {
  auto inputCount = node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");

  auto& inputs_0 = *ctx->input<Tensor>(0);

  // Ensure all of the non concatenated axes match each other
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->input<Tensor>(index);
    // Ensure all the other axes match
    auto dimensionCount = inputs_0.shape().NumDimensions();
    for (int axisIndex = 0; axisIndex < dimensionCount; axisIndex++) {
      if (axisIndex == axis_)
        continue;
      LOTUS_ENFORCE(data_n.shape()[axisIndex] == inputs_0.shape()[axisIndex], "Non concat axis dimensions must match");
    }
  }

  // Calculate the size of the concatenated axis, and verify all other dimensions match
  size_t concatAxisSize = 0;
  for (int index = 0; index < inputCount; index++) {
    concatAxisSize += ctx->input<Tensor>(index)->shape()[int(axis_)];
  }

  // Calculate the shape of the output tensor
  std::vector<int64_t> dims;
  for (int dimensionIndex = 0; dimensionIndex < inputs_0.shape().NumDimensions(); dimensionIndex++)
    dims.emplace_back(inputs_0.shape()[dimensionIndex]);
  dims[axis_] = concatAxisSize;
  TensorShape outputShape(dims);

  // The outputAxisPitch is the number of elements to add to move to the next split axis in the output
  int64_t outputAxisPitch = 1;
  for (auto i = int64_t(dims.size()); i-- > axis_;)
    outputAxisPitch *= dims[axis_];

  auto& concat_result = *ctx->output(0, outputShape);
  float* outputBase = concat_result.mutable_data<float>();

  for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
    auto& data_n = *ctx->input<Tensor>(inputIndex);

    // The inputAxisPitch is the number of elements to add to move to the next split axis in the input
    int64_t inputAxisPitch = 1;
    for (int i = int(data_n.shape().NumDimensions()); i-- > axis_;)
      inputAxisPitch *= data_n.shape()[int(i)];

    const float* input = data_n.data<float>();
    auto inputSize = data_n.shape().Size();

    // Copy the data across. For every 'inputAxisPitch' values copied, we move over by the 'outputAxisPitch'
    float* output = outputBase;
    for (int i = 0, j = 0; i < inputSize; i++) {
      output[i] = input[i];
      if (++j == inputAxisPitch) {
        output += outputAxisPitch - inputAxisPitch;  // Subtract inputAxisPitch because output is being indexed by 'i'
        j = 0;
      }
    }
    outputBase += inputAxisPitch;
  }
  return Status::OK();
}

}  // namespace Lotus
