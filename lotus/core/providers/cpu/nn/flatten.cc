#include "core/providers/cpu/nn/flatten.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("Flatten")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .MayInplace(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Flatten<float>);
}
