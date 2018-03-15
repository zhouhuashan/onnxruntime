#include "core/providers/cpu/math/clip.h"

namespace Lotus {

template <>
void Clip<float>::compute(OpKernelContext* ctx) {
  const Tensor* X = ctx->template input<Tensor>(0);
  Tensor* Y = ctx->output(0, X->shape());
  EigenVectorMap<float>(Y->mutable_data<float>(), Y->shape().Size()) =
      ConstEigenVectorMap<float>(X->data<float>(), X->shape().Size())
          .cwiseMax(min_)
          .cwiseMin(max_);
}
REGISTER_KERNEL(KernelDef("Clip")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Clip<float>);
}  // namespace Lotus
