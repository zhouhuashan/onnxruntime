#include "core/providers/cpu/tensor/pad.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Pad")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Pad<float>);

// This is the general padding method to n-dimensionally do edge or reflection padding (based on the inputDelta values)
template <typename T>
static void PadAxis(T *output, T *input, ptrdiff_t input_delta, ptrdiff_t input_pitch, size_t block_size, size_t block_count) {
  for (size_t block_index = 0; block_index < block_count; block_index++) {
    for (size_t i = 0; i < block_size; i++) {
      *output++ = *input;
      input += input_delta;
    }
    input += input_pitch;
  }
}

// These are optimizations of PadAxis. The inner loop is removed since the innermost axis has a blockSize of 1,
// and inputPitch and inputDelta are just a single value added each iteration.
template <typename T>
static void PadInnermostAxis(T *output, T *input, ptrdiff_t input_delta, size_t block_count) {
  for (size_t block_index = 0; block_index < block_count; block_index++) {
    *output++ = *input;
    input += input_delta;
  }
}

// For constant padding, there is no input, just a size to write the constant to
template <typename T>
static void PadAxisConstant(T *output, T constant, size_t size) {
  for (size_t i = 0; i < size; i++)
    *output++ = constant;
}

template <>
Status Pad<float>::Compute(OpKernelContext *ctx) const {
  auto &input_tensor = *ctx->Input<Tensor>(0);
  std::vector<int64_t> output_dims(input_tensor.Shape().GetDims());
  size_t dimension_count = output_dims.size();

  LOTUS_ENFORCE(dimension_count * 2 == pads_.size(), "'pads' attribute has wrong number of values");

  // Calculate output dimensions
  for (size_t i = 0; i < dimension_count; i++) {
    LOTUS_ENFORCE(pads_[i] >= 0 && pads_[i + dimension_count] >= 0, "Negative padding values are not allowed.");
    output_dims[i] += pads_[i] + pads_[i + dimension_count];
  }
  TensorShape output_shape(output_dims);

  const auto *input = input_tensor.Data<float>();
  auto &output_tensor = *ctx->Output(0, output_shape);
  auto *output = output_tensor.MutableData<float>();

  // The innerPitches is the pitch of the next inner axis. Aka the amount to move by one of the next inner axis.
  // For the innermost axis it's always 1, the next outer is the innermost axis size, etc.
  std::vector<int64_t> inner_pitches(dimension_count, 0);
  inner_pitches.back() = 1;
  for (size_t i = dimension_count - 1; i-- > 0;) {
    inner_pitches[i] = inner_pitches[i + 1] * output_dims[i + 1];
  }

  // Initial skip, sum up the begin padding on each axis (except the innermost)
  for (size_t i = 0; i < dimension_count - 1; i++)
    output += pads_[i] * inner_pitches[i];

  std::vector<int64_t> axis_index(dimension_count - 1, 0);  // There is no index for innermost axis since it's a special case
  bool running = true;

  switch (mode_) {
    case Mode::Constant:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (running) {
        size_t axis = dimension_count - 1;  // Start in the innermost axis
        output += pads_[axis];              // Skip over pre padding

        // Copy the input data over
        size_t size = input_tensor.Shape()[axis];
        for (size_t i = 0; i < size; i++)
          *output++ = *input++;

        // Write out padding
        PadAxisConstant(output - size - pads_[axis], value_, pads_[axis]);
        PadAxisConstant(output, value_, pads_[axis + dimension_count]);
        output += pads_[axis + dimension_count];

        if (axis == 0)  // If this is a 1D tensor, we're done
          break;

        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (++axis_index[--axis] == input_tensor.Shape()[axis]) {
          axis_index[axis] = 0;
          ptrdiff_t inner_pitch = inner_pitches[axis];
          float *outputStart = output - inner_pitch * input_tensor.Shape()[axis];
          PadAxisConstant(outputStart - pads_[axis] * inner_pitch, value_, pads_[axis] * inner_pitch);
          PadAxisConstant(output, value_, pads_[axis + dimension_count] * inner_pitch);
          output += inner_pitch * pads_[axis + dimension_count];
          if (axis == 0) {
            running = false;
            break;
          }
        }
      }
      break;

    case Mode::Edge:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (running) {
        size_t axis = dimension_count - 1;  // Start in the innermost axis
        output += pads_[axis];              // Skip over pre padding

        // Copy the input data over
        size_t size = input_tensor.Shape()[axis];
        for (size_t i = 0; i < size; i++)
          *output++ = *input++;

        PadInnermostAxis(output - size - pads_[axis], output - size, 0 /* inputDelta */, pads_[axis]);
        PadInnermostAxis(output, output - 1, 0 /* inputDelta */, pads_[axis + dimension_count]);
        output += pads_[axis + dimension_count];

        if (axis == 0)  // If this is a 1D tensor, we're done
          break;

        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (++axis_index[--axis] == input_tensor.Shape()[axis]) {
          ptrdiff_t inner_pitch = inner_pitches[axis];
          float *outputStart = output - inner_pitch * input_tensor.Shape()[axis];
          PadAxis(outputStart - pads_[axis] * inner_pitch, outputStart, 1, -inner_pitch, inner_pitch, pads_[axis]);
          PadAxis(output, output - inner_pitch, 1, -inner_pitch, inner_pitch, pads_[axis + dimension_count]);
          output += inner_pitch * pads_[axis + dimension_count];
          axis_index[axis] = 0;
          if (axis == 0) {
            running = false;
            break;
          }
        }
      }
      break;

    case Mode::Reflect:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (running) {
        size_t axis = dimension_count - 1;  // Start in the innermost axis
        output += pads_[axis];              // Skip over pre padding

        // Copy the input data over
        size_t size = input_tensor.Shape()[axis];
        for (size_t i = 0; i < size; i++)
          *output++ = *input++;

        PadInnermostAxis(output - size - pads_[axis], output - size + pads_[axis], -1 /* inputDelta */, pads_[axis]);
        PadInnermostAxis(output, output - pads_[axis + dimension_count], -1 /* inputDelta */, pads_[axis + dimension_count]);
        output += pads_[axis + dimension_count];

        if (axis == 0)  // If this is a 1D tensor, we're done
          break;

        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (++axis_index[--axis] == input_tensor.Shape()[axis]) {
          ptrdiff_t inner_pitch = inner_pitches[axis];
          float *outputStart = output - inner_pitch * input_tensor.Shape()[axis];
          PadAxis(outputStart - pads_[axis] * inner_pitch, outputStart + pads_[axis] * inner_pitch, 1, -inner_pitch * 2, inner_pitch, pads_[axis]);
          PadAxis(output, output - pads_[axis + dimension_count] * inner_pitch, 1, -inner_pitch * 2, inner_pitch, pads_[axis + dimension_count]);
          output += inner_pitch * pads_[axis + dimension_count];
          axis_index[axis] = 0;
          if (axis == 0) {
            running = false;
            break;
          }
        }
      }
      break;
  }

  return Status::OK();
}

};  // namespace Lotus
