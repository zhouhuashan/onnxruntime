#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/CUDA/Half.h"
#include "core/inc/op_kernel_author.h"
#include "core/util/math_cpuonly.h"

#if defined(USE_MLAS) && defined(_M_AMD64)
#include <windows.h>
#include <mlas.h>
#endif

namespace Lotus {

template <typename SrcType,
          typename DstType>
inline void CastData(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto shape_size = shape.Size();
  auto in_vector = ConstEigenVectorMap<SrcType>(in->Data<SrcType>(), shape_size);
  auto output_vector = EigenVectorMap<DstType>(out->MutableData<DstType>(), shape_size);
  output_vector = in_vector.template cast<DstType>();
}

template <>
inline void CastData<float, MLFloat16>(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto out_data = out->MutableData<MLFloat16>();
  auto shape_size = shape.Size();
  auto in_vector = ConstEigenVectorMap<float>(in->Data<float>(), shape_size);
  auto output_vector = EigenVectorMap<Eigen::half>(static_cast<Eigen::half*>(static_cast<void*>(out_data)), shape_size);
  output_vector = in_vector.template cast<Eigen::half>();
}

template <>
inline void CastData<MLFloat16, float>(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto out_data = out->MutableData<float>();
  auto in_data = in->Data<MLFloat16>();
  auto shape_size = shape.Size();
#if defined(USE_MLAS) && defined(_M_AMD64)
  MlasConvertHalfToFloatBuffer(&in_data[0].val, out_data, shape_size);
#else
  auto in_vector = ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(in_data)), shape_size);
  auto output_vector = EigenVectorMap<float>(out_data, shape_size);
  output_vector = in_vector.template cast<float>();
#endif
}

template <typename SrcType,
          typename DstType>
inline void CastFloat16Data(const Tensor* in, Tensor* out, const TensorShape& shape, const OpKernelInfo& info) {
  auto* p_provider = info.GetExecutionProvider();
  LOTUS_ENFORCE(p_provider);
  auto allocator = p_provider->GetAllocator();
  LOTUS_ENFORCE(allocator != nullptr);
  void* buffer = allocator->Alloc(sizeof(float) * shape.Size());
  LOTUS_ENFORCE(buffer);
  Tensor tmp_tensor(DataTypeImpl::GetType<float>(), shape, buffer, allocator->Info(), allocator);

  if (std::is_same<SrcType, MLFloat16>::value) {
    CastData<MLFloat16, float>(in, &tmp_tensor, shape);  // first cast to float
    CastData<float, DstType>(&tmp_tensor, out, shape);   // then cast to the destination type.
  } else if (std::is_same<DstType, MLFloat16>::value) {
    CastData<SrcType, float>(in, &tmp_tensor, shape);
    CastData<float, MLFloat16>(&tmp_tensor, out, shape);
  }
}

template <typename T>
class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    LOTUS_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<onnx::TensorProto_DataType>(to);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename SrcType,
            typename DstType>
  void CastData(const Tensor* in, Tensor* out, const TensorShape& shape) const {
    ::Lotus::CastData<SrcType, DstType>(in, out, shape);
  }

  template <typename SrcType,
            typename DstType>
  void CastFloat16Data(const Tensor* in, Tensor* out, const TensorShape& shape, const OpKernelInfo& info) const {
    ::Lotus::CastFloat16Data<SrcType, DstType>(in, out, shape, info);
  }

  onnx::TensorProto_DataType to_;
};

}  //namespace Lotus
