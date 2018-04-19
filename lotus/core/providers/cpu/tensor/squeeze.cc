#include "core/providers/cpu/tensor/squeeze.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("Squeeze")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .MayInplace(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Squeeze<float>);
}
