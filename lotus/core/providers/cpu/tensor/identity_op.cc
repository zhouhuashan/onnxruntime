#include "core/providers/cpu/tensor/identity_op.h"
#include "core/inc/op_kernel_author.h"

namespace Lotus {
	
ONNX_CPU_OPERATOR_KERNEL(
    Dropout,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    IdentityOp<true>);

ONNX_CPU_OPERATOR_KERNEL(
    Identity,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).Alias(0, 0),
    IdentityOp<false>);

}  // namespace Lotus
