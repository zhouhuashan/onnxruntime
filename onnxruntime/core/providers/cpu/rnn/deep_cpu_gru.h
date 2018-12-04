// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/common/task_thread_pool.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {

/// The class represents GRU operator using DeepCPU implementation for
/// fast inference computation on CPU machines.
template <typename T>
class DeepCpuGruOp final : public OpKernel {
 public:
  DeepCpuGruOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

  ~DeepCpuGruOp() override = default;

 private:
  Status TransposeWeight(const Tensor* W, const Tensor* R, AllocatorPtr& alloc);

  rnn::detail::Direction direction_;
  int num_directions_;

  int hidden_size_ = 0;
  float clip_;
  int linear_before_reset_ = 0;
  size_t hidden_size_3x_;
  size_t recurrent_weights_size_per_direction_;
  size_t bias_size_per_direction_;

  rnn::detail::ActivationFuncs activation_funcs_;

  // Whether transposed weights have been updated or not.
  bool is_weight_transposed = false;
  IAllocatorUniquePtr<T> weights_ptr_;
  gsl::span<T> input_weightsZRH_FW_, recurrent_weightsZR_FW_, recurrent_weightsH_FW_;
  gsl::span<T> input_weightsZRH_BW_, recurrent_weightsZR_BW_, recurrent_weightsH_BW_;

  // TO DO: move the thread to higher level
  // Threadpool for operator. If concurrent Compute calls are possible, it will be shared
  // across them. mutable due to this.
  // The alternative would be to create a threadpool in each call to Compute but that would incur thread creation
  // cost on every call.
  mutable TaskThreadPool ttp_{std::thread::hardware_concurrency()};
};

}  // namespace onnxruntime
