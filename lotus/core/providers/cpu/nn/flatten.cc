#include "core/providers/cpu/nn/flatten.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    Flatten,
    1,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);
}
