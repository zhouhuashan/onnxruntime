#include "core/providers/cpu/math/element_wise_ops.h"

namespace Lotus {

template <typename T>
auto EigenMap(Tensor& t) { return EigenVectorMap<T>(t.mutable_data<T>(), t.shape().Size()); }
template <typename T>
auto EigenMap(const Tensor& t) { return ConstEigenVectorMap<T>(t.data<T>(), t.shape().Size()); }

template <>
Status Add<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
  auto& C = *ctx->output(0, A.shape());

  EigenMap<float>(C) = EigenMap<float>(A) + EigenMap<float>(B);
  return Status::OK();
}

REGISTER_KERNEL(KernelDef("Add")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("A", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>()),
                Add<float>);

template <>
Status Sub<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
  auto& C = *ctx->output(0, A.shape());

  EigenMap<float>(C) = EigenMap<float>(A) - EigenMap<float>(B);
  return Status::OK();
}

REGISTER_KERNEL(KernelDef("Sub")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("A", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>()),
                Sub<float>);

template <>
Status Mul<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
  auto& C = *ctx->output(0, A.shape());

  EigenMap<float>(C) = EigenMap<float>(A).cwiseProduct(EigenMap<float>(B));
  return Status::OK();
}

REGISTER_KERNEL(KernelDef("Mul")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("A", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>()),
                Mul<float>);

template <>
Status Reciprocal<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseInverse();
  return Status::OK();
}

template <>
Status Sum<float>::compute(OpKernelContext* ctx) const {
  auto inputCount = node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->input<Tensor>(0);
  auto& shape = data_0.shape();
  auto sum = EigenMap<float>(*ctx->output(0, shape));

  if (inputCount == 1) {
    sum = EigenMap<float>(data_0);
    return Status::OK();
  }

  auto& data_1 = *ctx->input<Tensor>(1);
  LOTUS_ENFORCE(data_1.shape() == shape, "All inputs must have the same shape");

  sum = EigenMap<float>(data_0) + EigenMap<float>(data_1);
  for (int index = 2; index < inputCount; index++) {
    auto& data_n = *ctx->input<Tensor>(index);
    LOTUS_ENFORCE(data_n.shape() == shape, "All inputs must have the same shape");
    sum += EigenMap<float>(data_n);
  }

  return Status::OK();
}

}  // namespace Lotus
