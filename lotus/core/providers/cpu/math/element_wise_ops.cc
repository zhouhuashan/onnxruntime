#include "core/providers/cpu/math/element_wise_ops.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Add")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Add<float>);

REGISTER_KERNEL(KernelDefBuilder("Sub")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sub<float>);

REGISTER_KERNEL(KernelDefBuilder("Mul")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Mul<float>);

REGISTER_KERNEL(KernelDefBuilder("Div")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Div<float>);

REGISTER_KERNEL(KernelDefBuilder("Abs")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Abs<float>);

REGISTER_KERNEL(KernelDefBuilder("Neg")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Neg<float>);

REGISTER_KERNEL(KernelDefBuilder("Floor")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Floor<float>);

REGISTER_KERNEL(KernelDefBuilder("Ceil")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Ceil<float>);

REGISTER_KERNEL(KernelDefBuilder("Reciprocal")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Reciprocal<float>);

REGISTER_KERNEL(KernelDefBuilder("Sqrt")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sqrt<float>);

REGISTER_KERNEL(KernelDefBuilder("Pow")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Pow<float>);

REGISTER_KERNEL(KernelDefBuilder("Exp")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Exp<float>);

REGISTER_KERNEL(KernelDefBuilder("Log")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Log<float>);

REGISTER_KERNEL(KernelDefBuilder("Sum")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Sum<float>);

REGISTER_KERNEL(KernelDefBuilder("Min")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Min<float>);

REGISTER_KERNEL(KernelDefBuilder("Max")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Max<float>);

REGISTER_KERNEL(KernelDefBuilder("And")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
                And<bool>);

REGISTER_KERNEL(KernelDefBuilder("Or")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
                Or<bool>);

REGISTER_KERNEL(KernelDefBuilder("Xor")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
                Xor<bool>);

REGISTER_KERNEL(KernelDefBuilder("Less")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
                Less<float>);

REGISTER_KERNEL(KernelDefBuilder("Greater")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
                Greater<float>);

REGISTER_KERNEL(KernelDefBuilder("Equal")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
                Equal<float>);

template <typename T>
auto EigenMap(Tensor& t) { return EigenVectorMap<T>(t.MutableData<T>(), t.Shape().Size()); }
template <typename T>
auto EigenMap(const Tensor& t) { return ConstEigenVectorMap<T>(t.Data<T>(), t.Shape().Size()); }

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

template <typename TInput, typename Op>
void Loop(const Tensor& input1, const Tensor& input2, Tensor& output, Op op) {
  using TOutput = std::result_of_t<Op(TInput, TInput)>;

  const TInput* input1_data = input1.Data<TInput>();
  const TInput* input2_data = input2.Data<TInput>();
  TOutput* output_data = output.MutableData<TOutput>();
  auto outputSize = output.Shape().Size();

  for (auto i = 0; i < outputSize; i++)
    output_data[i] = op(input1_data[i], input2_data[i]);
}

template <typename T, typename Op>
void ScalarLoop(const Tensor& input1, T value, Tensor& output, Op op) {
  const T* input1_data = input1.Data<T>();
  T* output_data = output.MutableData<T>();
  auto outputSize = output.Shape().Size();

  for (auto i = 0; i < outputSize; i++)
    output_data[i] = op(input1_data[i], value);
}

template <typename T, typename Op>
void Broadcast(const Tensor& input1, const Tensor& input2, Tensor& output, int axis, Op op) {
  // If the axis_ attribute exists, use and verify it, otherwise look for the matching suffix
  if (axis == -1)
    axis = FindShapeSubsetAxis(input1.Shape(), input2.Shape());
  else
    VerifyShapeSubsetAxis(input1.Shape(), input2.Shape(), axis);

  // If the input tensor has dimensions like [2][3][4][5][6] and the second input has dimensions like [4][5]
  // Then we want to access the second as though the first two and last index is ignored, like this: [x][x][4][5][x] ('x' means value has no effect)
  // Since we're iterating sequentially through both tensors, we can do this by incrementing the index into
  // the second tensor every '2*3' elements (thus ignoring the first two dimensions),
  // and resetting the index every '2*3*4*5' elements (thus ignoring the last dimension)

  int64_t incrementPitch = 1;
  for (int i = int(input1.Shape().NumDimensions()); --i > axis;)
    incrementPitch *= input1.Shape()[i];

  int64_t resetPitch = input2.Shape().Size();

  const T* input1_data = input1.Data<T>();
  const T* input2_data = input2.Data<T>();
  T* output_data = output.MutableData<T>();
  auto outputSize = output.Shape().Size();

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
Status Add<float>::Compute(OpKernelContext* ctx) const {
  auto& A = *ctx->Input<Tensor>(0);
  auto& B = *ctx->Input<Tensor>(1);
  auto& C = *ctx->Output(0, A.Shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a + b; });
  else {
    LOTUS_ENFORCE(A.Shape() == B.Shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A) + EigenMap<float>(B);
  }

  return Status::OK();
}

template <>
Status Sub<float>::Compute(OpKernelContext* ctx) const {
  auto& A = *ctx->Input<Tensor>(0);
  auto& B = *ctx->Input<Tensor>(1);
  auto& C = *ctx->Output(0, A.Shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a - b; });
  else {
    LOTUS_ENFORCE(A.Shape() == B.Shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A) - EigenMap<float>(B);
  }

  return Status::OK();
}

template <>
Status Mul<float>::Compute(OpKernelContext* ctx) const {
  auto& A = *ctx->Input<Tensor>(0);
  auto& B = *ctx->Input<Tensor>(1);
  auto& C = *ctx->Output(0, A.Shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a * b; });
  else {
    LOTUS_ENFORCE(A.Shape() == B.Shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A).cwiseProduct(EigenMap<float>(B));
  }

  return Status::OK();
}

template <>
Status Div<float>::Compute(OpKernelContext* ctx) const {
  auto& A = *ctx->Input<Tensor>(0);
  auto& B = *ctx->Input<Tensor>(1);
  auto& C = *ctx->Output(0, A.Shape());

  if (broadcast_)
    Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return a / b; });
  else {
    LOTUS_ENFORCE(A.Shape() == B.Shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A).cwiseQuotient(EigenMap<float>(B));
  }

  return Status::OK();
}

template <>
Status Abs<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseAbs();

  return Status::OK();
}

template <>
Status Neg<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = -EigenMap<float>(X);

  return Status::OK();
}

template <>
Status Floor<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().floor();

  return Status::OK();
}

template <>
Status Ceil<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().ceil();

  return Status::OK();
}

template <>
Status Reciprocal<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseInverse();

  return Status::OK();
}

template <>
Status Sqrt<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseSqrt();

  return Status::OK();
}

template <>
Status Pow<float>::Compute(OpKernelContext* ctx) const {
  auto& A = *ctx->Input<Tensor>(0);
  auto& B = *ctx->Input<Tensor>(1);
  auto& C = *ctx->Output(0, A.Shape());

  if (broadcast_) {
    if (B.Shape().NumDimensions() == 0) {
      LOTUS_ENFORCE(axis_ == -1, "When broadcasting by a scalar, axis cannot be set");
      EigenMap<float>(C) = EigenMap<float>(A).array().pow(*B.Data<float>());
    } else
      Broadcast<float>(A, B, C, int(axis_), [](float a, float b) { return pow(a, b); });
  } else {
    LOTUS_ENFORCE(A.Shape() == B.Shape(), "Inputs must have the same shape");
    EigenMap<float>(C) = EigenMap<float>(A).array().pow(EigenMap<float>(B).array());
  }

  return Status::OK();
}

template <>
Status Exp<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().exp();

  return Status::OK();
}

template <>
Status Log<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().log();

  return Status::OK();
}

template <>
Status Sum<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto sum = EigenMap<float>(*ctx->Output(0, shape));

  if (inputCount == 1) {
    sum = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    LOTUS_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    sum = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < inputCount; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      LOTUS_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      sum += EigenMap<float>(data_n);
    }
  }

  return Status::OK();
}

template <>
Status Min<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto min = EigenMap<float>(*ctx->Output(0, shape));

  min = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    LOTUS_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    min = min.array().min(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <>
Status Max<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto max = EigenMap<float>(*ctx->Output(0, shape));

  max = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    LOTUS_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    max = max.array().max(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <typename TInput, typename Op>
Status BooleanOp(OpKernelContext* ctx, bool broadcast, int64_t axis, Op op) {
  auto& A = *ctx->Input<Tensor>(0);
  auto& B = *ctx->Input<Tensor>(1);
  auto& C = *ctx->Output(0, A.Shape());

  if (broadcast) {
    if (B.Shape().NumDimensions() == 0) {
      LOTUS_ENFORCE(axis == -1, "When broadcasting by a scalar, axis cannot be set");
      ScalarLoop<TInput>(A, *B.Data<TInput>(), C, op);
    } else
      Broadcast<TInput>(A, B, C, int(axis), op);
  } else {
    Loop<TInput>(A, B, C, op);
  }
  return Status::OK();
}

template <>
Status And<bool>::Compute(OpKernelContext* ctx) const {
  return BooleanOp<bool>(ctx, broadcast_, axis_, [](bool a, bool b) { return a && b; });
}

template <>
Status Or<bool>::Compute(OpKernelContext* ctx) const {
  return BooleanOp<bool>(ctx, broadcast_, axis_, [](bool a, bool b) { return a || b; });
}

template <>
Status Xor<bool>::Compute(OpKernelContext* ctx) const {
  return BooleanOp<bool>(ctx, broadcast_, axis_, [](bool a, bool b) { return (a ^ b) != 0; });
}

template <>
Status Equal<float>::Compute(OpKernelContext* ctx) const {
  return BooleanOp<float>(ctx, broadcast_, axis_, [](float a, float b) { return a == b; });
}

template <>
Status Less<float>::Compute(OpKernelContext* ctx) const {
  return BooleanOp<float>(ctx, broadcast_, axis_, [](float a, float b) { return a < b; });
}

template <>
Status Greater<float>::Compute(OpKernelContext* ctx) const {
  return BooleanOp<float>(ctx, broadcast_, axis_, [](float a, float b) { return a > b; });
}

}  // namespace Lotus
