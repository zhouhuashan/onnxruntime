#include "core/providers/cpu/tensor/space_depth_ops.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("SpaceToDepth")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                SpaceToDepth<float>);

REGISTER_KERNEL(KernelDefBuilder("DepthToSpace")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 4)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                DepthToSpace<float>);

template <>
Status SpaceToDepth<float>::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);
  LOTUS_ENFORCE(input.Shape().NumDimensions() == 4);
  const int64_t batch = input.Shape().GetDims().at(0);
  const int64_t input_depth = input.Shape().GetDims().at(1);
  const int64_t input_height = input.Shape().GetDims().at(2);
  const int64_t input_width = input.Shape().GetDims().at(3);
  LOTUS_ENFORCE(input_height % this->blocksize_ == 0);
  LOTUS_ENFORCE(input_width % this->blocksize_ == 0);

  const int64_t output_depth = input_depth * blocksize_ * blocksize_;
  const int64_t output_height = input_height / blocksize_;
  const int64_t output_width = input_width / blocksize_;
  std::vector<int64_t> output_dims({batch, output_depth, output_height, output_width});
  Tensor& output = *context->Output(0, output_dims);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t out_d = 0; out_d < output_depth; ++out_d) {
      const int64_t in_d = out_d % input_depth;
      const int64_t offset_w = (out_d / input_depth) % blocksize_;
      const int64_t offset_h = (out_d / input_depth) / blocksize_;
      for (int64_t out_h = 0; out_h < output_height; ++out_h) {
        const int64_t in_h = out_h * blocksize_ + offset_h;
        for (int64_t out_w = 0; out_w < output_width; ++out_w) {
          const int64_t in_w = out_w * blocksize_ + offset_w;
          const auto output_offset =
              ((b * output_depth + out_d) * output_height + out_h) *
                  output_width +
              out_w;
          const auto input_offset =
              ((b * input_depth + in_d) * input_height + in_h) *
                  input_width +
              in_w;
          if (in_h >= 0 && in_w >= 0 && in_h < input_height &&
              in_w < input_width) {
            output.template MutableData<float>()[output_offset] =
                input.template Data<float>()[input_offset];
          } else {
            output.template MutableData<float>()[output_offset] = 0.0;
          }
        }
      }
    }
  }

  return Status::OK();
}

template <>
Status DepthToSpace<float>::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);
  LOTUS_ENFORCE(input.Shape().NumDimensions() == 4);

  const int64_t batch = input.Shape().GetDims().at(0);
  const int64_t input_depth = input.Shape().GetDims().at(1);
  const int64_t input_height = input.Shape().GetDims().at(2);
  const int64_t input_width = input.Shape().GetDims().at(3);
  LOTUS_ENFORCE(input_depth % (blocksize_ * blocksize_) == 0);

  const int64_t output_depth = input_depth / blocksize_ / blocksize_;
  const int64_t output_height = input_height * blocksize_;
  const int64_t output_width = input_width * blocksize_;

  std::vector<int64_t> output_dims({batch, output_depth, output_height, output_width});
  Tensor& output = *context->Output(0, output_dims);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t in_d = 0; in_d < input_depth; ++in_d) {
      const int64_t out_d = in_d % output_depth;
      const int64_t offset_w = (in_d / output_depth) % blocksize_;
      const int64_t offset_h = (in_d / output_depth) / blocksize_;
      for (int64_t in_h = 0; in_h < input_height; ++in_h) {
        const int64_t out_h = in_h * blocksize_ + offset_h;
        for (int64_t in_w = 0; in_w < input_width; ++in_w) {
          const int64_t out_w = in_w * blocksize_ + offset_w;
          if (out_h >= 0 && out_w >= 0 && out_h < output_height &&
              out_w < output_width) {
            const auto output_offset =
                ((b * output_depth + out_d) * output_height + out_h) *
                    output_width +
                out_w;
            const auto input_offset =
                ((b * input_depth + in_d) * input_height + in_h) *
                    input_width +
                in_w;
            output.template MutableData<float>()[output_offset] =
                input.template Data<float>()[input_offset];
          }
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace Lotus
