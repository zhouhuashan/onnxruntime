#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

template <typename T>
class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<std::string> to;
    Status status = info.GetAttrs<std::string>("to", to);
    LOTUS_ENFORCE(status.IsOK(), "Attribute to is not set.");
    LOTUS_ENFORCE(to.size() == 1, "Attribute to can have only one Cast type.");
    LOTUS_ENFORCE(TensorProto_DataType_Parse(to[0], &to_), "Cast type: %s is not supported.", to[0]);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename SrcType,
            typename DstType>
  void CastData(const Tensor* in, Tensor* out, const TensorShape& shape) const {
    for (size_t i = 0; i < shape.Size(); ++i) {
      out->MutableData<DstType>()[i] = static_cast<DstType>(in->Data<SrcType>()[i]);
    }
  }

  TensorProto_DataType to_;
};
}  //namespace Lotus
