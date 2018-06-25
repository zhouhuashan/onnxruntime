#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/CUDA/Half.h"

namespace Lotus {
template <typename SrcType,
          typename DstType>
inline void CastData(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto out_data = out->MutableData<DstType>();
  auto in_data = in->Data<SrcType>();

  for (int64_t i = 0; i < shape.Size(); ++i) {
    out_data[i] = static_cast<DstType>(in_data[i]);
  }
}

template <>
inline void CastData<float, MLFloat16>(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto out_data = out->MutableData<MLFloat16>();
  auto in_data = in->Data<float>();
  for (int64_t i = 0; i < shape.Size(); ++i) {
    out_data[i] = MLFloat16(Eigen::half_impl::float_to_half_rtne(in_data[i]).x);
  }
}

template <>
inline void CastData<MLFloat16, float>(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto out_data = out->MutableData<float>();
  auto in_data = in->Data<MLFloat16>();
  for (int64_t i = 0; i < shape.Size(); ++i) {
    out_data[i] = Eigen::half_impl::half_to_float(Eigen::half_impl::__half(in_data[i].val));
  }
}

template <typename T>
class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    LOTUS_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<TensorProto_DataType>(to);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename SrcType,
            typename DstType>
  void CastData(const Tensor* in, Tensor* out, const TensorShape& shape) const {
    Lotus::CastData<SrcType, DstType>(in, out, shape);
  }

  TensorProto_DataType to_;
};
}  //namespace Lotus
