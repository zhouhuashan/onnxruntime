#include "core/providers/cpu/math/element_wise_ops.h"

namespace Lotus {

REGISTER_KERNEL(KernelDef("Add")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Add<float>);

REGISTER_KERNEL(KernelDef("Sub")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sub<float>);

REGISTER_KERNEL(KernelDef("Mul")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Mul<float>);

REGISTER_KERNEL(KernelDef("Div")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Div<float>);

REGISTER_KERNEL(KernelDef("Abs")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Abs<float>);

REGISTER_KERNEL(KernelDef("Neg")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Neg<float>);

REGISTER_KERNEL(KernelDef("Floor")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Floor<float>);

REGISTER_KERNEL(KernelDef("Ceil")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Ceil<float>);

REGISTER_KERNEL(KernelDef("Reciprocal")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Reciprocal<float>);

REGISTER_KERNEL(KernelDef("Sqrt")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sqrt<float>);

REGISTER_KERNEL(KernelDef("Pow")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Pow<float>);

REGISTER_KERNEL(KernelDef("Exp")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Exp<float>);

REGISTER_KERNEL(KernelDef("Log")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Log<float>);

REGISTER_KERNEL(KernelDef("Sum")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sum<float>);

REGISTER_KERNEL(KernelDef("Min")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(6, 6)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Min<float>);

REGISTER_KERNEL(KernelDef("Max")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(6, 6)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Max<float>);

template <typename T>
auto EigenMap(Tensor& t) { return EigenVectorMap<T>(t.mutable_data<T>(), t.shape().Size()); }
template <typename T>
auto EigenMap(const Tensor& t) { return ConstEigenVectorMap<T>(t.data<T>(), t.shape().Size()); }

// Finds the axis inside 'shape' that matches 'find' starting from the end
// For example if shape = {2, 3, 4, 5, 6} and find = {4, 5} it returns 2
// If shape = {4, 5, 2, 4, 5} and find = {4, 5} it would return 3
int FindShapeSubsetAxis(const TensorShape& shape, const TensorShape& find) {
  int findCount = int(find.NumDimensions());

  for (int i = int(shape.NumDimensions()) - findCount; i >= 0; i--) {
    int j = 0;
    for (; j < findCount; j++) {
      if (shape[i + j] != find[j])
        break;
    }
    if (j == findCount)
      return i;
  }
  LOTUS_THROW("Tensors have no common shape subset");
}

// Validate that 'find' matches 'shape' at location 'axis'
void VerifyShapeSubsetAxis(const TensorShape& shape, const TensorShape& find, int axis) {
  LOTUS_ENFORCE(axis >= 0 && axis < int(shape.NumDimensions()), "Axis attribute out of range");
  int dimensions = int(find.NumDimensions());
  for (int i = 0; i < dimensions; i++) {
    if (shape[int(axis) + i] != find[i])
      LOTUS_THROW("Axis attribute doesn't refer to a valid subset");
  }
}

template <typename T, typename Op>
void Broadcast(const Tensor& input1, const Tensor& input2, Tensor& output, int axis, Op op) {
  // If the axis_ attribute exists, use and verify it, otherwise look for the matching suffix
  if (axis == -1)
    axis = FindShapeSubsetAxis(input1.shape(), input2.shape());
  else
    VerifyShapeSubsetAxis(input1.shape(), input2.shape(), axis);

  // If the input tensor has dimensions like [2][3][4][5][6] and the second input has dimensions like [4][5]
  // Then we want to access the second as though the first two and last index is ignored, like this: [x][x][4][5][x] ('x' means value has no effect)
  // Since we're iterating sequentially through both tensors, we can do this by incrementing the index into
  // the second tensor every '2*3' elements (thus ignoring the first two dimensions),
  // and resetting the index every '2*3*4*5' elements (thus ignoring the last dimension)

  int64_t incrementPitch = 1;
  for (int i = int(input1.shape().NumDimensions()); --i > axis;)
    incrementPitch *= input1.shape()[i];

  int64_t resetPitch = input2.shape().Size();

  const T* input1_data = input1.data<T>();
  const T* input2_data = input2.data<T>();
  float* output_data = output.mutable_data<T>();
  auto outputSize = output.shape().Size();

  // Do the operation
  int input2_index = 0;
  int incrementCount = 0;
  for (int i = 0; i < outputSize; i++) {
    *output_data++ = op(*input1_data++, input2_data[input2_index]);

    if (++incrementCount == incrementPitch) {
      incrementCount = 0;
      if (++input2_index == resetPitch) {
        input2_index = 0;
      }
    }
  }
}

template <>
Status Add<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  auto& C = *ctx->output(0, A.shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a + b; });
  else {
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A) + EigenMap<float>(B);
  }
  return Status::OK();
}

template <>
Status Sub<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  auto& C = *ctx->output(0, A.shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a - b; });
  else {
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A) - EigenMap<float>(B);
  }
  return Status::OK();
}

template <>
Status Mul<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  auto& C = *ctx->output(0, A.shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a * b; });
  else {
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A).cwiseProduct(EigenMap<float>(B));
  }
  return Status::OK();
}

template <>
Status Div<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  auto& C = *ctx->output(0, A.shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a / b; });
  else {
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A).cwiseQuotient(EigenMap<float>(B));
  }
  return Status::OK();
}

template <>
Status Abs<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseAbs();
  return Status::OK();
}

template <>
Status Neg<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = -EigenMap<float>(X);
  return Status::OK();
}

template <>
Status Floor<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().floor();
  return Status::OK();
}

template <>
Status Ceil<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().ceil();
  return Status::OK();
}

template <>
Status Reciprocal<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseInverse();
  return Status::OK();
}

template <>
Status Sqrt<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseSqrt();
  return Status::OK();
}

template <>
Status Pow<float>::compute(OpKernelContext* ctx) const {
  auto& A = *ctx->input<Tensor>(0);
  auto& B = *ctx->input<Tensor>(1);
  auto& C = *ctx->output(0, A.shape());

  if (broadcast_) {
    if (B.shape().NumDimensions() == 0) {
      LOTUS_ENFORCE(axis_ == -1, "When broadcasting by a scalar, axis cannot be set");
      EigenMap<float>(C) = EigenMap<float>(A).array().pow(*B.data<float>());
    } else
      Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return pow(a, b); });
  } else {
    LOTUS_ENFORCE(A.shape() == B.shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A).array().pow(EigenMap<float>(B).array());
  }
  return Status::OK();
}

template <>
Status Exp<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().exp();
  return Status::OK();
}

template <>
Status Log<float>::compute(OpKernelContext* ctx) const {
  auto& X = *ctx->input<Tensor>(0);
  auto& Y = *ctx->output(0, X.shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().log();
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

template <>
Status Min<float>::compute(OpKernelContext* ctx) const {
  auto inputCount = node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->input<Tensor>(0);
  auto& shape = data_0.shape();
  auto min = EigenMap<float>(*ctx->output(0, shape));

  min = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->input<Tensor>(index);
    LOTUS_ENFORCE(data_n.shape() == shape, "All inputs must have the same shape");
    min = min.array().min(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <>
Status Max<float>::compute(OpKernelContext* ctx) const {
  auto inputCount = node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->input<Tensor>(0);
  auto& shape = data_0.shape();
  auto max = EigenMap<float>(*ctx->output(0, shape));

  max = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->input<Tensor>(index);
    LOTUS_ENFORCE(data_n.shape() == shape, "All inputs must have the same shape");
    max = max.array().max(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

}  // namespace Lotus
