#ifndef CORE_PROVIDERS_CPU_MATH_ELEMENT_WISE_OPS_H
#define CORE_PROVIDERS_CPU_MATH_ELEMENT_WISE_OPS_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

class BroadcastAxisKernel : public OpKernel {
 protected:
  BroadcastAxisKernel(const OpKernelInfo& info) : OpKernel(info) {
    int64_t broadcast;
    broadcast_ = info.GetAttr("broadcast", &broadcast).IsOK() && broadcast == 1;
    info.GetAttr("axis", &axis_).IsOK();
    LOTUS_ENFORCE(axis_ == -1 || axis_ != -1 && broadcast_, "If 'axis' attribute is specified, then 'broadcast' attribute should be set to one.");
  }

  bool broadcast_;
  int64_t axis_{-1};  // -1 means 'no axis specified'
};

template <typename T>
class Add final : public BroadcastAxisKernel {
 public:
  Add(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  void compute(OpKernelContext* context) override;
};

template <typename T>
class Sub final : public BroadcastAxisKernel {
 public:
  Sub(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  void compute(OpKernelContext* context) override;
};

template <typename T>
class Mul final : public BroadcastAxisKernel {
 public:
  Mul(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  void compute(OpKernelContext* context) override;
};

template <typename T>
class Reciprocal final : public OpKernel {
 public:
  Reciprocal(const OpKernelInfo& info) : OpKernel(info) {
  }

  void compute(OpKernelContext* context) override;
};

template <typename T>
class Sum final : public OpKernel {
 public:
  Sum(const OpKernelInfo& info) : OpKernel(info) {
  }

  void compute(OpKernelContext* context) override;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_MATH_ELEMENT_WISE_OPS_H
