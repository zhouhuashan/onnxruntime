#include "core/providers/cpu/math/clip.h"

namespace Lotus {

ONNX_CPU_OPERATOR_KERNEL(
    Clip,
    6,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Clip<float>);

}  // namespace Lotus
