#include "core/providers/cpu/activation/activations.h"

namespace Lotus {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(alias, x, sinceVersion)           \
  REGISTER_KERNEL(KernelDefBuilder(#alias)                                        \
                      .Domain(LotusIR::kOnnxDomain)                               \
                      .SinceVersion(sinceVersion)                                 \
                      .Provider(LotusIR::kCpuExecutionProvider)                   \
                      .MayInplace(0, 0)                                           \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
                  x<float>)

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion) \
  REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(x, x, sinceVersion)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ParametricSoftplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ScaledTanh, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 1);
// SoftPlus is the default case for ParametricSoftPlus
REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(Softplus, ParametricSoftplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(PRelu, 6);

template <typename T>
auto EigenMap(Tensor& t) { return EigenVectorMap<T>(t.MutableData<T>(), t.Shape().Size()); }
template <typename T>
auto EigenMap(const Tensor& t) { return ConstEigenVectorMap<T>(t.Data<T>(), t.Shape().Size()); }

bool SupportedBroadcast(const TensorShape& slope_shape, const TensorShape& input_shape, size_t& num_images) {
  // currently only supports slope being in channel only, and regard any leading dims as num_images beyond slope_shape
  // TODO: change the PReLU implementation to support generic numpy broadcast per ONNX 1.2.1 spec
  size_t input_dims = input_shape.NumDimensions();
  size_t slope_dims = slope_shape.NumDimensions();
  size_t padded_dims = input_dims - slope_dims;
  if (input_dims < slope_dims) return false;

  // only support single non-broadcast (dim > 1) channel as for legacy PReLU usage
  int channel_idx = -1;
  for (int idx = (int)slope_dims - 1; idx >= 0; --idx) {
    if (slope_shape[idx] > 1) {
      // channel need to match between input_shape and slope_shape
      if (input_shape[padded_dims + idx] != slope_shape[idx])
        return false;
      // only support single channel
      if (channel_idx != -1) return false;
      channel_idx = idx;
    }
  }

  num_images = 1;
  for (size_t idx = 0; idx < padded_dims + channel_idx; idx++) {
    num_images *= input_shape[idx];
  }
  return true;
}

template <>
Status PRelu<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& slope = *ctx->Input<Tensor>(1);
  auto& Y = *ctx->Output(0, X.Shape());

  auto eigenX = EigenMap<float>(X).array();
  auto eigenY = EigenMap<float>(Y);
  if (slope.Shape().IsScalar()) {
    eigenY = eigenX.cwiseMax(0.0f) + eigenX.cwiseMin(0.0f) * *slope.Data<float>();
  } else if (slope.Shape().Size() != X.Shape().Size()) {
    size_t num_images;
    LOTUS_ENFORCE(SupportedBroadcast(slope.Shape(), X.Shape(), num_images));
    int64_t num_channels = slope.Shape().Size();
    int64_t image_size = X.Shape().Size() / num_images;
    int64_t num_pixels = image_size / num_channels;
    for (size_t image = 0; image < num_images; image++) {
      for (int64_t channel = 0; channel < num_channels; channel++) {
        auto segY = eigenY.segment(image * image_size + num_pixels * channel, num_pixels);
        auto segX = eigenX.segment(image * image_size + num_pixels * channel, num_pixels);
        segY = segX.cwiseMax(0.0f) + segX.cwiseMin(0.0f) * slope.Data<float>()[channel];
      }
    }
  } else {
    LOTUS_ENFORCE(X.Shape() == slope.Shape(), "Inputs must have the same shape if slope is not a scalar");
    eigenY = eigenX.cwiseMax(0.0f) + eigenX.cwiseMin(0.0f) * EigenMap<float>(slope).array();
  }
  return Status::OK();
}

}  // namespace Lotus
