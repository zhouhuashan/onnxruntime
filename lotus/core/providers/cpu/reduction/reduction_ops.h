#ifndef CORE_PROVIDERS_CPU_REDUCTION_OPS_H
#define CORE_PROVIDERS_CPU_REDUCTION_OPS_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

class ReduceKernel : public OpKernel {
 protected:
  ReduceKernel(const OpKernelInfo& info, bool allow_multi_axes = true) : OpKernel(info) {
    if (allow_multi_axes) {
      LOTUS_ENFORCE(info.GetAttrs("axes", axes_).IsOK());
    } else {
      axes_.resize(1);
      LOTUS_ENFORCE(info.GetAttr("axis", &axes_[0]).IsOK());
    }
    int64_t keepdims = 1;
    if (info.GetAttr("keepdims", &keepdims).IsOK())
      keepdims_ = (keepdims == 1);
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
class ReduceLogSum final : public ReduceKernel {
 public:
  ReduceLogSum(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceLogSumExp final : public ReduceKernel {
 public:
  ReduceLogSumExp(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceMax final : public ReduceKernel {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceMean final : public ReduceKernel {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceMin final : public ReduceKernel {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel(info) {
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

template <typename T>
class ReduceSum final : public ReduceKernel {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReduceSumSquare final : public ReduceKernel {
 public:
  ReduceSumSquare(const OpKernelInfo& info) : ReduceKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ArgMax final : public ReduceKernel {
 public:
  ArgMax(const OpKernelInfo& info) : ReduceKernel(info, false) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ArgMin final : public ReduceKernel {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel(info, false) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_REDUCTION_OPS_H
