#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace Lotus {
namespace Cuda {

struct BinaryElementwisePreparation {
  const Tensor* lhs_tensor = nullptr;
  const Tensor* rhs_tensor = nullptr;
  Tensor* output_tensor = nullptr;
  size_t output_rank_or_simple_broadcast = 0;  // for no_broadcast|left_scalar|right_scalar cases, output_rank uses SimpleBroadcast enums
  bool lhs_dim0_broadcast = false;
  bool rhs_dim0_broadcast = false;
  CudaAsyncBuffer<int64_t> lhs_padded_strides;
  CudaAsyncBuffer<int64_t> rhs_padded_strides;
  CudaAsyncBuffer<fast_divmod> fdm_output_strides;

  BinaryElementwisePreparation(CUDAExecutionProvider* provider) : lhs_padded_strides(provider),
                                                                  rhs_padded_strides(provider),
                                                                  fdm_output_strides(provider) {}
};

// trait classes to indicate if the kernel supports broadcast
class ShouldBroadcast {
};

class ShouldNotBroadcast {
};

template <typename BroadcastTrait>
class BinaryElementwise : public CudaKernel {
 protected:
  typedef BroadcastTrait broadcast_type;

  BinaryElementwise(const OpKernelInfo& info) : CudaKernel(info) {}
  Status Compute(OpKernelContext*) const override {
    return Status(Common::LOTUS, Common::FAIL);  // should not reach here
  }
  Status Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const;
};

template <typename T>
class Add final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Add(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Sub(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Mul(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Div(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Pow final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Pow(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class And final : public BinaryElementwise<ShouldBroadcast> {
 public:
  And(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Or final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Or(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Xor final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Xor(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status Compute(OpKernelContext* context) const override;
};

// PRelu is activation function, but it's closer to binary elementwise ops in implementation
template <typename T>
class PRelu final : public BinaryElementwise<ShouldBroadcast> {
 public:
  PRelu(const OpKernelInfo& info) : BinaryElementwise(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

// Sum allows varadic inputs, and it uses binary elementwise Add in implementation
template <typename T>
class Sum final : public CudaKernel {
 public:
  Sum(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace Cuda
}  // namespace Lotus
