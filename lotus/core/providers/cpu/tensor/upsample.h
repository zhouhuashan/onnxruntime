#pragma once

#include "core/framework/op_kernel.h"

namespace Lotus {

template <typename T>
class Upsample : public OpKernel {
 public:
  Upsample(OpKernelInfo info) : OpKernel(info) {
    std::string mode;
    if (!info.GetAttr<std::string>("mode", &mode).IsOK())
      mode = Upsample<float>::UpsampleModeNN;
    mode_ = StringToUpsampleMode(mode);

    LOTUS_ENFORCE(info.GetAttrs<float>("scales", scales_).IsOK());
    for (auto& scale : scales_) {
      LOTUS_ENFORCE(scale >= 1, "Scale value should be greater than or equal to 1.");
    }
  }

  virtual ~Upsample() = default;

  Status Compute(OpKernelContext* context) const;

 protected:
  static const std::string UpsampleModeNN;
  static const std::string UpsampleModeLinear;

  enum class UpsampleMode {
    NN = 0,      // nearest neighbour
    LINEAR = 1,  // linear interpolation
  };

  UpsampleMode mode_;

  std::vector<float> scales_;

  UpsampleMode StringToUpsampleMode(const std::string& mode) {
    if (mode == Upsample<float>::UpsampleModeNN) {
      return UpsampleMode::NN;
    } else if (mode == Upsample<float>::UpsampleModeLinear) {
      return UpsampleMode::LINEAR;
    } else {
      LOTUS_THROW("mode attribute is " + mode + ". It can only be " +
                  UpsampleModeNN + "(default) or " + UpsampleModeLinear + ".");
    }
  }
};
}  // namespace Lotus
