#include "core/providers/cpu/tensor/image_scaler.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("ImageScaler")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ImageScaler<float>);
}  // namespace Lotus
