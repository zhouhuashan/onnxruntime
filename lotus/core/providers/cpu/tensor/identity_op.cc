#include "core/providers/cpu/tensor/identity_op.h"

namespace Lotus {
//copying reshape kernel
REGISTER_KERNEL(KernelDef("Dropout")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                IdentityOp<float>);
}  // namespace Lotus
