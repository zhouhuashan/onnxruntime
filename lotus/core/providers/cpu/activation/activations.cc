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

template <>
Status PRelu<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& slope = *ctx->Input<Tensor>(1);
  auto& Y = *ctx->Output(0, X.Shape());

  auto eigenX = EigenMap<float>(X).array();
  auto eigenY = EigenMap<float>(Y);
  if (slope.Shape().IsScalar()) {
    eigenY = eigenX.cwiseMax(0.0f) + eigenX.cwiseMin(0.0f) * *slope.Data<float>();
  } else if (slope.Shape().NumDimensions() == 1) {
    int64_t num_channels = slope.Shape()[0];
    int64_t input_dims = X.Shape().NumDimensions();
    LOTUS_ENFORCE(input_dims > 1 && num_channels == X.Shape()[1]);
    size_t num_images = X.Shape()[0];
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
