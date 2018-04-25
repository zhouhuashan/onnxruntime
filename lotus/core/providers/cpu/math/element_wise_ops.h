#pragma once

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

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public BroadcastAxisKernel {
 public:
  Sub(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public BroadcastAxisKernel {
 public:
  Mul(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public BroadcastAxisKernel {
 public:
  Div(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Abs final : public OpKernel {
 public:
  Abs(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Neg final : public OpKernel {
 public:
  Neg(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Floor final : public OpKernel {
 public:
  Floor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Ceil final : public OpKernel {
 public:
  Ceil(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Reciprocal final : public OpKernel {
 public:
  Reciprocal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sqrt final : public OpKernel {
 public:
  Sqrt(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Pow final : public BroadcastAxisKernel {
 public:
  Pow(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Exp final : public BroadcastAxisKernel {
 public:
  Exp(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Log final : public BroadcastAxisKernel {
 public:
  Log(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum final : public OpKernel {
 public:
  Sum(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Min final : public OpKernel {
 public:
  Min(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Max final : public OpKernel {
 public:
  Max(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class And final : public BroadcastAxisKernel {
 public:
  And(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Or final : public BroadcastAxisKernel {
 public:
  Or(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Xor final : public BroadcastAxisKernel {
 public:
  Xor(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Equal final : public BroadcastAxisKernel {
 public:
  Equal(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Less final : public BroadcastAxisKernel {
 public:
  Less(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Greater final : public BroadcastAxisKernel {
 public:
  Greater(const OpKernelInfo& info) : BroadcastAxisKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mean final : public OpKernel {
 public:
  Mean(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Affine final : public OpKernel {
 public:
  Affine(const OpKernelInfo& info) : OpKernel(info) {
    alpha_ = 1.0f;  // default value; TODO : can be omitted once added to ONNX spec.
    beta_ = 0.0f;   // default value; TODO : can be omitted once added to ONNX spec.
    info.GetAttr("alpha", &alpha_);
    info.GetAttr("beta", &beta_);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
};

}  // namespace Lotus
