#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace Lotus {
namespace Cuda {

class ReduceKernel : public CudaKernel {
 protected:
  ReduceKernel(const OpKernelInfo& info, bool allow_multi_axes = true) : CudaKernel(info) {
    if (allow_multi_axes) {
      info.GetAttrs("axes", axes_);
    } else {
      axes_.push_back(0);
      info.GetAttr("axis", &axes_[0]);
    }
    int64_t keepdims;
    info.GetAttrOrDefault("keepdims", &keepdims, (int64_t)1);
    keepdims_ = (keepdims == 1);
  }

  ~ReduceKernel() {}

  template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES>
  Status ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnnReduceOp) const;

 private:
  std::vector<int64_t> axes_;
  bool keepdims_ = true;
};

template <typename T>
class ArgMax final : public ReduceKernel {
 public:
  ArgMax(const OpKernelInfo& info) : ReduceKernel(info, false) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES>(ctx, CUDNN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ArgMin final : public ReduceKernel {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel(info, false) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES>(ctx, CUDNN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceL1 final : public ReduceKernel {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_NORM1);
  }
};

template <typename T>
class ReduceL2 final : public ReduceKernel {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_NORM2);
  }
};

template <typename T>
class ReduceMax final : public ReduceKernel {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ReduceMean final : public ReduceKernel {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_AVG);
  }
};

template <typename T>
class ReduceMin final : public ReduceKernel {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceProd final : public ReduceKernel {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MUL);
  }
};

template <typename T>
class ReduceSum final : public ReduceKernel {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
  }
};

}  // namespace Cuda
}  // namespace Lotus
