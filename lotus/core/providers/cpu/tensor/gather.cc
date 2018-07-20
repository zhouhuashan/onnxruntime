#include "core/providers/cpu/tensor/gather.h"

namespace Lotus {
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Gather,
    1,
	int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                      .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    Gather<float, int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Gather,
    1,
	int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                      .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int32_t>()),
    Gather<float, int32_t>);
}  // namespace Lotus
