#include "core/providers/cpu/activation/tanh.h"

namespace Lotus {
template class Tanh<float>;
REGISTER_KERNEL(KernelDef("Tanh")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Tanh<float>);
}  // namespace Lotus
