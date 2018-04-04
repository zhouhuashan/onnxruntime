#ifndef CORE_PROVIDERS_CPU_REDUCTION_OPS_H
#define CORE_PROVIDERS_CPU_REDUCTION_OPS_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

class ReduceKernel : public OpKernel {
 protected:
  ReduceKernel(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(info.GetAttrs("axes", axes_).IsOK());
    int64_t keepdims = 1;
    if (info.GetAttr("keepdims", &keepdims).IsOK())
      keepdims_ = keepdims == 1;
  }

  void PrepareForReduce(OpKernelContext* ctx, std::vector<float>& transposedInputData, Tensor** reducedTensor,
                        int64_t& block_size, int64_t& blocks) const;

  std::vector<int64_t> axes_;
  bool keepdims_ = true;
};

template <typename T>
class ReduceL1 final : public ReduceKernel {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceL2 final : public ReduceKernel {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceProd final : public ReduceKernel {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_REDUCTION_OPS_H
