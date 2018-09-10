#include "core/providers/cpu/tensor/reshape.h"
#include "core/inc/op_kernel_author.h"
namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Reshape,
    5,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape_1,
    1,
    4,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Reshape_1);

}  // namespace onnxruntime
