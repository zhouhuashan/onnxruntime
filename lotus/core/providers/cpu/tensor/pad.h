#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T>
struct Pad final : OpKernel {
  Pad(const OpKernelInfo& info) : OpKernel(info) {
    std::string mode;
    if (info.GetAttr("mode", &mode).IsOK()) {
      if (mode == "constant")
        mode_ = Mode::Constant;
      else if (mode == "reflect")
        mode_ = Mode::Reflect;
      else if (mode == "edge")
        mode_ = Mode::Edge;
      else
        LOTUS_THROW("Invalid 'mode' attribute value");
    }
    if (!info.GetAttrs("pads", pads_).IsOK())
      LOTUS_THROW("Invalid 'pads' attribute value");

    // Separate out any negative pads_ into the slices_ array
    slices_.resize(pads_.size(), 0);
    for (size_t index = 0; index < pads_.size(); index++) {
      if (pads_[index] < 0) {
        slices_[index] = pads_[index];
        pads_[index] = 0;
      }
    }

    info.GetAttr("value", &value_);  // Value is optional and initialized to 0 by default
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  enum class Mode {
    Constant,
    Reflect,
    Edge
  };
  Mode mode_{Mode::Constant};
  std::vector<int64_t> pads_;    // After construction, only >=0 values are in here
  std::vector<int64_t> slices_;  // All of the negative padding values are separated out into slices_
  T value_{};
};

}  // namespace Lotus
