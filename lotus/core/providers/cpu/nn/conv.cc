#include "core/providers/cpu/nn/conv.h"

namespace Lotus {
REGISTER_KERNEL(KernelDef("Conv")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("X", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("W", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>()),
                Conv<float>);
}
