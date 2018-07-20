#include "core/providers/cpu/tensor/crop.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    Crop,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Crop<float>);
}  // namespace Lotus
