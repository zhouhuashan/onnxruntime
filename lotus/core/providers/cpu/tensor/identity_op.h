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

    const T* source = X->Data<T>();
    T* target = Y->MutableData<T>();
    //If source and target pointers are not equal, we need to copy the data.
    if (target != source) {
      memcpy(target, source, shape.Size() * sizeof(T));
    }

    return Status::OK();
  }
};

}  //namespace Lotus
