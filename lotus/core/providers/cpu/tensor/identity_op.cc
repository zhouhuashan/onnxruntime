#include "core/providers/cpu/tensor/identity_op.h"
#include "core/inc/op_kernel_author.h"

namespace Lotus {
//copying reshape kernel
REGISTER_KERNEL(KernelDefBuilder("Dropout")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(7)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
                IdentityOp<true>);

REGISTER_KERNEL(KernelDefBuilder("Identity")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .Alias(0, 0)
                    .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
                IdentityOp<false>);

}  // namespace Lotus
