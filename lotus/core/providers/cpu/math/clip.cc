#include "core/providers/cpu/math/clip.h"

namespace Lotus {

template <>
Status Clip<float>::compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->template input<Tensor>(0);
  Tensor* Y = ctx->output(0, X->shape());
  EigenVectorMap<float>(Y->mutable_data<float>(), Y->shape().Size()) =
      ConstEigenVectorMap<float>(X->data<float>(), X->shape().Size())
          .cwiseMax(min_)
          .cwiseMin(max_);
  return Status::OK();
}
REGISTER_KERNEL(KernelDef("Clip")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Clip<float>);
}  // namespace Lotus
