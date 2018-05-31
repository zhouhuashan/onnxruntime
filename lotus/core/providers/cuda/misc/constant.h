#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "../cuda_common.h"

namespace Lotus {
namespace Cuda {

struct Constant final : public CudaKernel {
  Constant(const OpKernelInfo& info) : CudaKernel(info) {
    LOTUS_ENFORCE(info.GetAttr("value", &value_).IsOK(), "Must have valid 'value' attribute");
  }
  Status Compute(OpKernelContext* context) const override;

 private:
  TensorProto value_;
};

}  // namespace Cuda
}  // namespace Lotus
