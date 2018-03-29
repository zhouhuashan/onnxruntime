#ifndef CORE_PROVIDERS_CPU_NN_CONV_BASE_H
#define CORE_PROVIDERS_CPU_NN_CONV_BASE_H
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/autopad_type.h"

namespace Lotus {

// base class used by Conv and ConvTranspose
class ConvBase : public OpKernel {
 public:
  ConvBase(const OpKernelInfo& info) : OpKernel(info) {
    std::string auto_pad;
    auto status = info.GetAttr<std::string>("auto_pad", &auto_pad);
    auto_pad_ = status.IsOK() ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET;

    LOTUS_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK());

    status = info.GetAttrs<int64_t>("strides", strides_);
    if (!status.IsOK()) {
      strides_.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttrs<int64_t>("pads", pads_);
    if (!status.IsOK()) {
      pads_.resize(kernel_shape_.size() * 2, 0);
    }

    status = info.GetAttrs<int64_t>("dilations", dilations_);
    if (!status.IsOK()) {
      dilations_.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttr<int64_t>("group", &group_);
    if (!status.IsOK()) {
      group_ = 1;
    }

#if false
    // TODO: Re-enable when attributes values are guaranteed to be filled.
    std::string auto_pad;
    LOTUS_ENFORCE(info.GetAttr<std::string>("auto_pad", &auto_pad).IsOK());
    auto_pad_ = StringToAutoPadType(auto_pad);
    LOTUS_ENFORCE(info.GetAttr<int64_t>("group", &group_).IsOK());
    LOTUS_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK());
    LOTUS_ENFORCE(info.GetAttrs<int64_t>("strides", strides_).IsOK());
    LOTUS_ENFORCE(info.GetAttrs<int64_t>("pads", pads_).IsOK());
    LOTUS_ENFORCE(info.GetAttrs<int64_t>("dilations", dilations_).IsOK());
#endif
  }

  virtual ~ConvBase() {}

 protected:
  AutoPadType auto_pad_;
  int64_t group_;
  vector<int64_t> kernel_shape_;
  vector<int64_t> strides_;
  vector<int64_t> pads_;
  vector<int64_t> dilations_;
};

}  // namespace Lotus

#endif  // CORE_PROVIDERS_CPU_NN_CONV_BASE_H
