#include "core/providers/cpu/tensor/image_scaler.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    ImageScaler,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ImageScaler<float>);
}  // namespace Lotus
