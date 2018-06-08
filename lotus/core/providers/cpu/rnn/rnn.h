#pragma once

//#include "core/common/common.h"
//#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
template <typename T>
class RNN : public OpKernel {
  const set<string> allowed_activations{"Relu", "Tanh", "Sigmoid", "Affine", "LeakyRelu", "ThresholdedRelu", "ScaledTanh", "HardSigmoid", "Elu", "Softsign", "Softplus"};
  const string default_activation = "Tanh";
  const set<string> allowed_directions{"forward", "reverse", "bidirectional"};
  const string default_direction = "forward";

 public:
  RNN(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrs("activation_alpha", activation_alpha_);
    info.GetAttrs("activation_beta", activation_beta_);
    info.GetAttrs("activations", activations_);
    info.GetAttr("clip", &clip_);
    info.GetAttr("direction", &direction_);
    LOTUS_ENFORCE(info.GetAttr("hidden_size", &hidden_size_).IsOK());

    LOTUS_ENFORCE(allowed_directions.find(direction_) != allowed_directions.end());
    int num_directions = direction_ == "bidirectional" ? 2 : 1;

    // assign default attributes
    if (activation_alpha_.empty())
      activation_alpha_ = vector<float>(num_directions, 0.0F);
    if (activation_beta_.empty())
      activation_beta_ = vector<float>(num_directions, 0.0F);
    if (activations_.empty())
      activations_ = vector<string>(num_directions, default_activation);
    else if (activations_.size() == 2 && num_directions == 1) {
      // ONNX RNN default activations are {"Tanh", "Tanh"}
      // In this case, take the first default activation.
      activations_.resize(1);
    }

    LOTUS_ENFORCE(activations_.size() == num_directions);
    for (int direction = 1; direction < num_directions; direction++) {
      LOTUS_ENFORCE(allowed_activations.find(activations_[direction]) != allowed_activations.end());
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  // optional, default values tied to the activation function
  vector<float> activation_alpha_;

  // optional, default values tied to the activation function
  vector<float> activation_beta_;

  // optional, default = "Tanh"
  vector<string> activations_;

  // optional, default no clip_
  float clip_ = -1;

  // optional
  string direction_ = default_direction;

  // required
  int64_t hidden_size_;

  // const std::string default_activation = "Tanh";
};

}  // namespace Lotus
