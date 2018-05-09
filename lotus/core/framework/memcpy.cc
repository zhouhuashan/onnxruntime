#include "memcpy.h"

namespace Lotus {

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
    .Input(0, "X", "input", "T")
    .Output(0, "Y", "output", "T")
    .TypeConstraint(
        "T",
        OpSchema::all_tensor_types(),
        "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
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
    .SetDoc(R"DOC(
Internal copy node
)DOC");

REGISTER_KERNEL(KernelDefBuilder("MemcpyFromHost")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .HostMemory(0, true)
                    .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
                Memcpy);

REGISTER_KERNEL(KernelDefBuilder("MemcpyToHost")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .HostMemory(0, false)
                    .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
                Memcpy);

Memcpy::Memcpy(const OpKernelInfo& info)
    : OpKernel(info) {
  provider_ = info.GetExecutionProvider();
}

Status Memcpy::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());
  return provider_->CopyTensor(*X, *Y);
}

}  // namespace Lotus
