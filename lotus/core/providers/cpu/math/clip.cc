#include "core/providers/cpu/math/clip.h"

namespace Lotus {

template <>
Status Clip<float>::compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->template input<Tensor>(0);
  Tensor* Y = ctx->output(0, X->shape());
  if (has_max_ && has_min_) {
    EigenVectorMap<float>(Y->mutable_data<float>(), Y->shape().Size()) =
        ConstEigenVectorMap<float>(X->data<float>(), X->shape().Size())
            .cwiseMax(min_)
            .cwiseMin(max_);
  } else if (has_max_) {
    EigenVectorMap<float>(Y->mutable_data<float>(), Y->shape().Size()) =
        ConstEigenVectorMap<float>(X->data<float>(), X->shape().Size())
            .cwiseMin(max_);
  } else if (has_min_) {
    EigenVectorMap<float>(Y->mutable_data<float>(), Y->shape().Size()) =
        ConstEigenVectorMap<float>(X->data<float>(), X->shape().Size())
            .cwiseMax(min_);
  } else {
    //Copy input to output
    memcpy(Y->mutable_data<float>(), X->data<float>(), X->shape().Size());
  }
  return Status::OK();
}

REGISTER_KERNEL(KernelDef("Clip")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .MayInplace(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Clip<float>);
}  // namespace Lotus
