#pragma once

#include <limits>

#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/lib/task_thread_pool.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace Lotus {

/// The class represents GRU operator using DeepCPU implementation for
/// fast inference computation on CPU machines.
class DeepCpuGruOp final : public OpKernel {
 public:
  DeepCpuGruOp(const OpKernelInfo& info) : OpKernel(info) {
    // required attributes
    std::string direction;
    LOTUS_ENFORCE(info.GetAttr("direction", &direction).IsOK());

    int64_t int64_value;
    LOTUS_ENFORCE(info.GetAttr("linear_before_reset", &int64_value).IsOK());
    linear_before_reset_ = gsl::narrow<int>(int64_value);

    // TODO: Implementation needs changes to support linear_before_reset as Rbh and Wbh are added together
    // at the start of processing in the current version and are not available as separate values where ht
    // is calculated. VSTS Task 609: Support linear_before_reset in GRU operator
    LOTUS_ENFORCE(linear_before_reset_ == 0, "linear_before_reset is not currently supported.");

    LOTUS_ENFORCE(info.GetAttr("hidden_size", &int64_value).IsOK() && int64_value > 0);
    hidden_size_ = gsl::narrow<int>(int64_value);

    // optional attributes
    std::vector<std::string> activation_func_names;
    std::vector<float> activation_func_alphas;
    std::vector<float> activation_func_betas;
    info.GetAttrs<std::string>("activations", activation_func_names);
    info.GetAttrs<float>("activation_alpha", activation_func_alphas);
    info.GetAttrs<float>("activation_beta", activation_func_betas);

    info.GetAttr<float>("clip", &clip_);
    LOTUS_ENFORCE(clip_ > 0.f);

    direction_ = Rnn::detail::MakeDirection(direction);
    num_directions_ = direction_ == Rnn::detail::Direction::kBidirectional ? 2 : 1;

    if (activation_func_names.empty()) {
      for (int i = 0; i < num_directions_; ++i) {
        activation_func_names.emplace_back("sigmoid");
        activation_func_names.emplace_back("tanh");
      }
    }

    LOTUS_ENFORCE(activation_func_names.size() == num_directions_ * 2);

    activation_funcs_ = Rnn::detail::ActivationFuncs(activation_func_names,
                                                     activation_func_alphas,
                                                     activation_func_betas);
  }

  Status Compute(OpKernelContext* context) const override;

  ~DeepCpuGruOp() override = default;

 private:
  Rnn::detail::Direction direction_;
  int num_directions_;

  int hidden_size_ = 0;
  float clip_ = std::numeric_limits<float>::max();
  int linear_before_reset_ = 0;

  Rnn::detail::ActivationFuncs activation_funcs_;

  // Threadpool for operator. If concurrent Compute calls are possible, it will be shared
  // across them. mutable due to this.
  // The alternative would be to create a threadpool in each call to Compute but that would incur thread creation
  // cost on every call.
  mutable TaskThreadPool ttp_{std::thread::hardware_concurrency()};

  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;
};

}  // namespace Lotus
