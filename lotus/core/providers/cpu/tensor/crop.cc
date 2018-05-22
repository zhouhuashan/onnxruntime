#include "core/providers/cpu/tensor/crop.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Crop")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Crop<float>);
}  // namespace Lotus
