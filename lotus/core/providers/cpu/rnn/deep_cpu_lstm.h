#pragma once

#include <limits>

#include "core/framework/op_kernel.h"
#include "core/lib/task_thread_pool.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace Lotus {

/// The class represents DeepCPU implementation of a long short term memory (LSTM) operator.
/// For details, refer to http://aka.ms/dl-optimization/.
class DeepCpuLstmOp final : public OpKernel {
 public:
  DeepCpuLstmOp(const OpKernelInfo& info)
      : OpKernel(info) {
    std::string direction;
    LOTUS_ENFORCE(op_kernel_info_.GetAttr("direction", &direction).IsOK());

    int64_t int64_value;
    LOTUS_ENFORCE(op_kernel_info_.GetAttr("hidden_size", &int64_value).IsOK() && int64_value > 0);
    hidden_size_ = gsl::narrow<int>(int64_value);

    // optional attributes
    vector<std::string> activation_func_names;
    vector<float> activation_func_alphas;
    vector<float> activation_func_betas;
    op_kernel_info_.GetAttrs<std::string>("activations", activation_func_names);
    op_kernel_info_.GetAttrs<float>("activation_alpha", activation_func_alphas);
    op_kernel_info_.GetAttrs<float>("activation_beta", activation_func_betas);

    op_kernel_info_.GetAttr<float>("clip", &clip_);
    LOTUS_ENFORCE(clip_ > 0.f);

    if (op_kernel_info_.GetAttr("input_forget", &int64_value).IsOK())
      input_forget_ = int64_value != 0;

    direction_ = Rnn::detail::MakeDirection(direction);
    num_directions_ = direction_ == Rnn::detail::Direction::kBidirectional ? 2 : 1;

    if (activation_func_names.empty()) {
      for (int i = 0; i < num_directions_; ++i) {
        activation_func_names.push_back("sigmoid");
        activation_func_names.push_back("tanh");
        activation_func_names.push_back("tanh");
      }
    }

    LOTUS_ENFORCE(activation_func_names.size() == num_directions_ * 3);

    activation_funcs_ = Rnn::detail::ActivationFuncs(activation_func_names,
                                                     activation_func_alphas,
                                                     activation_func_betas);
  }

  Status Compute(OpKernelContext* context) const override;

  ~DeepCpuLstmOp() = default;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;

  Status ValidateInputs(const Tensor& X,
                        const Tensor& W,
                        const Tensor& R,
                        const Tensor* B,
                        const Tensor* sequence_lens,
                        const Tensor* initial_h,
                        const Tensor* initial_c,
                        const Tensor* P,
                        int batch_size) const;

  Rnn::detail::Direction direction_;
  int num_directions_;

  int hidden_size_ = 0;
  float clip_ = std::numeric_limits<float>::max();
  bool input_forget_ = false;

  Rnn::detail::ActivationFuncs activation_funcs_;

  // Threadpool for operator. If concurrent Compute calls are possible, it will be shared
  // across them. mutable due to this.
  // The alternative would be to create a threadpool in each call to Compute but that would incur thread creation
  // cost on every call.
  mutable TaskThreadPool ttp_{std::thread::hardware_concurrency()};
};

}  // namespace Lotus
