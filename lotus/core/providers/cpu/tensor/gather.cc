//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather
#include "core/providers/cpu/tensor/gather.h"
#include "core/inc/op_kernel_author.h"
#include "core/common/common.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    Gather,
    1,
    KernelDefBuilder().TypeConstraint("T", std::vector<MLDataType>{
                                               DataTypeImpl::GetTensorType<int32_t>(),
                                               DataTypeImpl::GetTensorType<float>()})
                      .TypeConstraint("Tind", std::vector<MLDataType>{
                                               DataTypeImpl::GetTensorType<int32_t>(),
                                               DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

Status Gather::Compute(OpKernelContext* context) const {
  const Tensor& T = *context->Input<Tensor>(0);
  const Tensor& Tind = *context->Input<Tensor>(1);

  MLDataType T_type = T.DataType();
  MLDataType Tind_type = Tind.DataType();

  if (Tind_type == DataTypeImpl::GetType<int32_t>()) {
    return TindTypedGatherImpl<int32_t>(context, T_type);
  } else if (Tind_type == DataTypeImpl::GetType<int64_t>()) {
    return TindTypedGatherImpl<int64_t>(context, T_type);
  } else {
    return LOTUS_MAKE_STATUS(LOTUS, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
  }
}

template <typename Tind_type>
Status Gather::TindTypedGatherImpl(OpKernelContext* context, const MLDataType& T_type) const {
  if (T_type == DataTypeImpl::GetType<int32_t>()) {
    return GatherImpl<int32_t, Tind_type>(context);
  } else if (T_type == DataTypeImpl::GetType<float>()) {
    return GatherImpl<float, Tind_type>(context);
  } else {
    return LOTUS_MAKE_STATUS(LOTUS, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
  }
}

}  // namespace Lotus
