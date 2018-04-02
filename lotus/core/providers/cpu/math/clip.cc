#include "core/providers/cpu/math/clip.h"

namespace Lotus {

template <>
Status Clip<float>::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());

  if (has_max_ && has_min_) {
    EigenVectorMap<float>(Y->MutableData<float>(), Y->Shape().Size()) =
        ConstEigenVectorMap<float>(X->Data<float>(), X->Shape().Size())
            .cwiseMax(min_)
            .cwiseMin(max_);
  } else if (has_max_) {
    EigenVectorMap<float>(Y->MutableData<float>(), Y->Shape().Size()) =
        ConstEigenVectorMap<float>(X->Data<float>(), X->Shape().Size())
            .cwiseMin(max_);
  } else if (has_min_) {
    EigenVectorMap<float>(Y->MutableData<float>(), Y->Shape().Size()) =
        ConstEigenVectorMap<float>(X->Data<float>(), X->Shape().Size())
            .cwiseMax(min_);
  } else {
    //Copy input to output
    memcpy(Y->MutableData<float>(), X->Data<float>(), X->Shape().Size());
  }

  return Status::OK();
}

REGISTER_KERNEL(KernelDefBuilder("Clip")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .MayInplace(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Clip<float>);

}  // namespace Lotus
