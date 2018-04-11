#include "core/providers/cpu/activation/activations.h"

namespace Lotus {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion)                        \
  REGISTER_KERNEL(KernelDefBuilder(#x)                                            \
                      .Domain(LotusIR::kOnnxDomain)                               \
                      .SinceVersion(sinceVersion)                                 \
                      .Provider(LotusIR::kCpuExecutionProvider)                   \
                      .MayInplace(0, 0)                                           \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
                  x<float>)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
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
  if (slope.Shape().NumDimensions() == 0) {
    eigenY = eigenX.cwiseMax(0.0f) + eigenX.cwiseMin(0.0f) * *slope.Data<float>();
  } else {
    LOTUS_ENFORCE(X.Shape() == slope.Shape(), "Inputs must have the same shape if slope is not a scalar");
    eigenY = eigenX.cwiseMax(0.0f) + eigenX.cwiseMin(0.0f) * EigenMap<float>(slope).array();
  }
  return Status::OK();
}

}  // namespace Lotus
