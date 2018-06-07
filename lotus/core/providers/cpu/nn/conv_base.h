#pragma once

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

    kernel_shape_specified_ = info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK();

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

  ~ConvBase() override {}

 protected:
  vector<int64_t> ComputeKernelShape(const TensorShape& weight_shape) const {
    if (kernel_shape_specified_)
      return kernel_shape_;
    else {
      auto& weight_dims = weight_shape.GetDims();
      vector<int64_t> result(weight_dims.begin() + 2, weight_dims.end());
      return result;
    }
  }

  AutoPadType auto_pad_;
  int64_t group_;
  bool kernel_shape_specified_;
  vector<int64_t> strides_;
  vector<int64_t> pads_;
  vector<int64_t> dilations_;

 private:
  vector<int64_t> kernel_shape_;  // must use ComputeKernelShape(...), instead of kernel_shape_
};

}  // namespace Lotus
