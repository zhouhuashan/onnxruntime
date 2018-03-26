#include "core/providers/cpu/activation/sigmoid.h"

namespace Lotus {

template class Sigmoid<float>;
REGISTER_KERNEL(KernelDef("Sigmoid")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sigmoid<float>);
}  // namespace Lotus
