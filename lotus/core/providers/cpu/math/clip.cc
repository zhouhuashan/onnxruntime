#include "core/providers/cpu/math/clip.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("Clip")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .MayInplace(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Clip<float>);

}  // namespace Lotus
