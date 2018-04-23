#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "gsl/gsl_util"

namespace Lotus {
template <typename T>
class Flatten final : public OpKernel {
 public:
  Flatten(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();
    LOTUS_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis_, "The rank of input tensor must be >= axis");

    Tensor* Y = context->Output(0, TensorShape({X_shape.SizeToDimension(axis_), X_shape.SizeFromDimension(axis_)}));
    //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    const T* source = X->Data<T>();
    T* target = Y->MutableData<T>();
    if (target != source) {
      memcpy(target, source, X_shape.Size() * sizeof(T));
    }

    return Status::OK();
  }

 private:
  int64_t axis_;
};
}  // namespace Lotus
