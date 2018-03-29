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

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public BroadcastAxisKernel {
 public:
  Sub(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public BroadcastAxisKernel {
 public:
  Mul(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public BroadcastAxisKernel {
 public:
  Div(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Abs final : public OpKernel {
 public:
  Abs(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Neg final : public OpKernel {
 public:
  Neg(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Floor final : public OpKernel {
 public:
  Floor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Ceil final : public OpKernel {
 public:
  Ceil(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Reciprocal final : public OpKernel {
 public:
  Reciprocal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Sqrt final : public OpKernel {
 public:
  Sqrt(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Pow final : public BroadcastAxisKernel {
 public:
  Pow(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Exp final : public BroadcastAxisKernel {
 public:
  Exp(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Log final : public BroadcastAxisKernel {
 public:
  Log(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum final : public OpKernel {
 public:
  Sum(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Min final : public OpKernel {
 public:
  Min(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Max final : public OpKernel {
 public:
  Max(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_MATH_ELEMENT_WISE_OPS_H
