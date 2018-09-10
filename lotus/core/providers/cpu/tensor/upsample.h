#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

constexpr const char* UpsampleModeNN = "nearest";
constexpr const char* UpsampleModeLinear = "linear";

template <typename T>
class Upsample : public OpKernel {
 public:
  Upsample(OpKernelInfo info) : OpKernel(info) {
    std::string mode;
    LOTUS_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());

    mode_ = StringToUpsampleMode(mode);

    LOTUS_ENFORCE(info.GetAttrs<float>("scales", scales_).IsOK());
    for (auto& scale : scales_) {
      LOTUS_ENFORCE(scale >= 1, "Scale value should be greater than or equal to 1.");
    }
  }

  ~Upsample() override = default;

  Status Compute(OpKernelContext* context) const override;

 protected:
  enum class UpsampleMode {
    NN = 0,      // nearest neighbour
    LINEAR = 1,  // linear interpolation
  };

  UpsampleMode mode_;

  std::vector<float> scales_;

  UpsampleMode StringToUpsampleMode(const std::string& mode) {
    if (strcmp(mode.c_str(), UpsampleModeNN) == 0) {
      return UpsampleMode::NN;
    } else if (strcmp(mode.c_str(), UpsampleModeLinear) == 0) {
      return UpsampleMode::LINEAR;
    } else {
      LOTUS_THROW("mode attribute is " + mode + ". It can only be " +
                  UpsampleModeNN + "(default) or " + UpsampleModeLinear + ".");
    }
  }
};
}  // namespace onnxruntime
