#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

template <typename T>
class IdentityOp final : public OpKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& shape = X->Shape();
    Tensor* Y = context->Output(0, TensorShape(shape));

    for (int64_t i = 0; i < shape.Size(); ++i) {
      Y->MutableData<T>()[i] = X->Data<T>()[i];
    }

    return Status::OK();
  }
};

}  //namespace Lotus
