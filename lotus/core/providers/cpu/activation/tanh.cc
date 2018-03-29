#include "core/providers/cpu/activation/tanh.h"

namespace Lotus {
template class Tanh<float>;
REGISTER_KERNEL(KernelDefBuilder("Tanh")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Tanh<float>);
}  // namespace Lotus
