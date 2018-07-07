#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace Lotus {
namespace Cuda {

template <typename SrcT>
class Cast final : public CudaKernel {
 public:
  Cast(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    LOTUS_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<onnx::TensorProto_DataType>(to);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  onnx::TensorProto_DataType to_;
};

}  // namespace Cuda
}  //namespace Lotus
