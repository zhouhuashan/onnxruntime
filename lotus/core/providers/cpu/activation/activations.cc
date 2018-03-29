#include "core/providers/cpu/activation/activations.h"

namespace Lotus {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x)                                      \
  REGISTER_KERNEL(KernelDef(#x)                                                   \
                      .Domain(LotusIR::kOnnxDomain)                               \
                      .SinceVersion(1, 2)                                         \
                      .Provider(LotusIR::kCpuExecutionProvider)                   \
                      .MayInplace(0, 0)                                           \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
                  x<float>)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu);

}  // namespace Lotus
