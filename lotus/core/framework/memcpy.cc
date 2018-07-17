#include "memcpy.h"
using namespace onnx;
namespace Lotus {

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
    .Input(0, "X", "input", "T")
    .Output(0, "Y", "output", "T")
    .TypeConstraint(
        "T",
        OpSchema::all_tensor_types(),
        "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
    .SetDoc(R"DOC(
Internal copy node
)DOC");

ONNX_OPERATOR_SCHEMA(MemcpyToHost)
    .Input(0, "X", "input", "T")
    .Output(0, "Y", "output", "T")
    .TypeConstraint(
        "T",
        OpSchema::all_tensor_types(),
        "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
    .SetDoc(R"DOC(
Internal copy node
)DOC");

Memcpy::Memcpy(const OpKernelInfo& info)
    : OpKernel(info) {
  provider_ = info.GetExecutionProvider();
}

Status Memcpy::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());
  Status retval = provider_->CopyTensor(*X, *Y, op_kernel_info_.GetKernelDef().ExecQueueId());
  return retval;
}

}  // namespace Lotus
