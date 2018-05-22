#include "core/providers/cpu/tensor/upsample.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("Upsample")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(7)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Upsample<float>);

void upsampleNearest2x(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    const float* input,
    float* output) {
  const int64_t output_height = input_height * 2;
  const int64_t output_width = input_width * 2;
  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        const int64_t in_y = y / 2;
        for (int64_t x = 0; x < input_width; ++x) {
          const float v = input[in_y * input_width + x];
          const int64_t oidx = output_width * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
      }
      input += input_height * input_width;
      output += output_height * output_width;
    }
  }
}

void upsampleNearest(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    float height_scale,
    float width_scale,
    const float* Xdata,
    float* Ydata) {
  // Specialized implementation for fast 2x upsampling
  if (width_scale == 2.0 && height_scale == 2.0) {
    upsampleNearest2x(
        batch_size, num_channels, input_height, input_width, Xdata, Ydata);
  }

  int64_t output_width = static_cast<int64_t>(input_width * width_scale);
  int64_t output_height = static_cast<int64_t>(input_height * height_scale);

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        const int64_t in_y = std::min(static_cast<int64_t>(y / height_scale), input_height - 1);
        for (int64_t x = 0; x < output_width; ++x) {
          const int64_t in_x = std::min(static_cast<int64_t>(x / width_scale), input_width - 1);
          Ydata[output_width * y + x] = Xdata[input_width * in_y + in_x];
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }
}

void upsampleBilinear(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    float height_scale,
    float width_scale,
    const float* Xdata,
    float* Ydata) {
  int64_t output_width = static_cast<int64_t>(input_width * width_scale);
  int64_t output_height = static_cast<int64_t>(input_height * height_scale);

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        float in_y = std::min(y / height_scale, static_cast<float>(input_height - 1));
        const int64_t in_y1 = std::min(static_cast<int64_t>(in_y), input_height - 1);
        const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        float dy1 = std::abs(in_y - in_y1);
        float dy2 = std::abs(in_y - in_y2);
        if (in_y1 == in_y2) {
          dy1 = 0.5f;
          dy2 = 0.5f;
        }

        for (int64_t x = 0; x < output_width; ++x) {
          float in_x = std::min(x / width_scale, static_cast<float>(input_width - 1));
          const int64_t in_x1 = std::min(static_cast<int64_t>(in_x), input_width - 1);
          const int64_t in_x2 = std::min(in_x1 + 1, input_width - 1);

          float dx1 = std::abs(in_x - in_x1);
          float dx2 = std::abs(in_x - in_x2);
          if (in_x1 == in_x2) {
            dx1 = 0.5f;
            dx2 = 0.5f;
          }

          float X11 = Xdata[input_width * in_y1 + in_x1];
          float X21 = Xdata[input_width * in_y1 + in_x2];
          float X12 = Xdata[input_width * in_y2 + in_x1];
          float X22 = Xdata[input_width * in_y2 + in_x2];

          Ydata[output_width * y + x] = dx2 * dy2 * X11 +
                                        dx1 * dy2 * X21 +
                                        dx2 * dy1 * X12 +
                                        dx1 * dy1 * X22;
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }
}

template <>
Status Upsample<float>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const std::vector<int64_t>& dims = X->Shape().GetDims();
  if (dims.size() != 4) {
    return Status(LOTUS, INVALID_ARGUMENT, "Upsample only support 2D inputs");
  }

  const int64_t batch_size = dims[0], num_channels = dims[1];
  const int64_t input_height = dims[2], input_width = dims[3];

  std::vector<int64_t> Y_dims({batch_size, num_channels,
                               (int64_t)(input_height * scales_[0]), (int64_t)(input_width * scales_[1])});
  Tensor* Y = context->Output(0, Y_dims);

  switch (mode_) {
    case UpsampleMode::NN:
      upsampleNearest(batch_size, num_channels, input_height, input_width,
                      scales_[0], scales_[1], X->Data<float>(), Y->MutableData<float>());
      break;
    case UpsampleMode::LINEAR:
      upsampleBilinear(batch_size, num_channels, input_height, input_width,
                       scales_[0], scales_[1], X->Data<float>(), Y->MutableData<float>());
  }

  return Status::OK();
}

}  // namespace Lotus
