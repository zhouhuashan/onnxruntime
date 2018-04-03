#include "core/providers/cpu/nn/conv.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("Conv")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Conv<float>);
}
