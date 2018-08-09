#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "gsl/gsl_util"

namespace Lotus {

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
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    if (target != source) {
      auto is_string_type = (X->DataType() == DataTypeImpl::GetType<std::string>());
      if (is_string_type) {
        for (int64_t i = 0; i < X->Shape().Size(); ++i)
          static_cast<std::string*>(target)[i] = static_cast<const std::string*>(source)[i];
      } else {
        memcpy(target, source, X_shape.Size() * X->DataType()->Size());
      }
    }

    return Status::OK();
  }

 private:
  int64_t axis_;
};
}  // namespace Lotus
