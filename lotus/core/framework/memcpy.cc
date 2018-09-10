#include "memcpy.h"
using namespace onnx;
namespace onnxruntime {

Memcpy::Memcpy(const OpKernelInfo& info)
    : OpKernel(info) {
  provider_ = info.GetExecutionProvider();
}

Status Memcpy::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());
  Status retval = provider_->CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());
  return retval;
}

}  // namespace onnxruntime
