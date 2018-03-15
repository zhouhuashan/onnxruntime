#include "core/providers/cpu/activation/sigmoid.h"

namespace Lotus {

template class Sigmoid<float>;
REGISTER_KERNEL(KernelDef("Sigmoid")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sigmoid<float>);
}  // namespace Lotus
