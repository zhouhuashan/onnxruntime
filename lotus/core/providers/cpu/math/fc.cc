#include "core/providers/cpu/math/fc.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("FC")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                FC<float, float, float, float>);
}
