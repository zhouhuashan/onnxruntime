#include "core/providers/cpu/nn/flatten.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    Flatten,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Flatten<float>);
}
