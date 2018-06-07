#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "core/common/common.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "core/framework/op_kernel.h"

namespace Lotus {

class IdentityOp final : public OpKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& shape = X->Shape();
    Tensor* Y = context->Output(0, TensorShape(shape));
    auto X_type = X->DataType();

    const void* source = X->DataRaw(X_type);
    void* target = Y->MutableDataRaw(X_type);
    //If source and target pointers are not equal, we need to copy the data.
    if (target != source) {
      if (X_type != DataTypeImpl::GetType<std::string>()) {
        memcpy(target, source, shape.Size() * X_type->Size());
      } else {
        // handle std::string
        const std::string* src = X->Data<std::string>();
        std::string* dst = Y->MutableData<std::string>();
        std::copy(src, src + shape.Size(), dst);
      }
    }

    return Status::OK();
  }
};

}  //namespace Lotus
