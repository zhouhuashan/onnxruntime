#include "core/providers/cpu/math/element_wise_ops.h"

namespace Lotus {

ONNX_CPU_OPERATOR_KERNEL(
    Add,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Add<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Sub,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sub<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Mul,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mul<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Div,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Div<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Abs,
    6,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Abs<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Abs,
    6,
    int8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    Abs<int8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Abs,
    6,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Abs<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Neg<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    int8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    Neg<int8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Neg<int32_t>);

ONNX_CPU_OPERATOR_KERNEL(
    Floor,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Floor<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Ceil,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Ceil<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Reciprocal,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Reciprocal<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Sqrt,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sqrt<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Pow,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pow<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Exp,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Exp<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Log,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Log<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Sum,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sum<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Min,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Min<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Max,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Max<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Not,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Not);

ONNX_CPU_OPERATOR_KERNEL(
    And,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    And);

ONNX_CPU_OPERATOR_KERNEL(
    Or,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Or);

ONNX_CPU_OPERATOR_KERNEL(
    Xor,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Xor);

ONNX_CPU_OPERATOR_KERNEL(
    Less,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Less<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Greater,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Greater<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Equal,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Equal<int32_t>);

ONNX_CPU_OPERATOR_KERNEL(
    Mean,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mean<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Affine,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Affine<float>);

template <typename T>
auto MakeEigenArrayMap(Tensor& t) { return EigenVectorArrayMap<T>(t.MutableData<T>(), t.Shape().Size()); }
template <typename T>
auto MakeEigenArrayMap(const Tensor& t) { return ConstEigenVectorArrayMap<T>(t.Data<T>(), t.Shape().Size()); }

struct BroadcastIterator {
  size_t AdvanceBy(size_t delta) {
    size_t index = index_;

    index_ += deltas_[0] * delta;
    counters_[0] += delta;
    if (counters_[0] == counts_[0]) {
      counters_[0] = 0;
      for (size_t counterIndex = 1; counterIndex < counters_.size(); counterIndex++) {
        index_ += deltas_[counterIndex];
        if (++counters_[counterIndex] != counts_[counterIndex])
          break;
        counters_[counterIndex] = 0;
      }
    }
    return index;
  }

  void Init(int64_t axis, int64_t largest) {
    LOTUS_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

    deltas_.push_back(axis > 1);
    counts_.push_back(largest);
    count_ *= axis;
  }

  void Append(int64_t axis, int64_t largest) {
    LOTUS_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

    // If we're greater than 1, it doesn't matter what the other tensor does
    if (axis > 1) {
      if (deltas_.back() <= 0)  // Were we broadcasting
        StopBroadcasting();
    } else {  // We must be 1, at this point
      if (deltas_.back() > 0)
        StartBroadcasting();
    }

    counts_.back() *= largest;  // Just increase the last count
    count_ *= axis;
  }

  void StopBroadcasting() {
    deltas_.push_back(count_);
    counts_.push_back(1);
  }

  void StartBroadcasting() {
    deltas_.push_back(-count_);
    counts_.push_back(1);
  }

  std::vector<int64_t> counters_;
  std::vector<ptrdiff_t> deltas_;
  std::vector<int64_t> counts_;
  size_t count_{1};  // Running total count of entries in tensor, used while building up the entries

 private:
  size_t index_{};
};

struct Broadcaster {
  Broadcaster(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    size_t dimension_count_max = std::max(shape1.size(), shape2.size());
    size_t dimension_count_min = std::min(shape1.size(), shape2.size());
    output_shape_.resize(dimension_count_max);

    auto iter1 = shape1.end();
    auto iter2 = shape2.end();
    auto output_shape = output_shape_.end();

    // Scalars are a special case, as it's always a broadcast
    size_t index = 0;
    if (dimension_count_min == 0) {
      if (shape1.size() == 0)  // Shape1 is a scalar
      {
        if (shape2.size() == 0)  // Two scalars?
        {
          iterator1_.Init(1, 1);
          iterator2_.Init(1, 1);
        } else {
          auto axis = *--iter2;
          iterator1_.Init(1, axis);
          iterator2_.Init(axis, axis);
          *--output_shape = axis;
        }
      } else {  // Shape2 is a scalar
        auto axis = *--iter1;
        iterator1_.Init(axis, axis);
        iterator2_.Init(1, axis);
        *--output_shape = axis;
      }
      index++;  // Manually increment since we processed one axis
    }

    for (; index < dimension_count_min; index++) {
      auto axis1 = *--iter1;
      auto axis2 = *--iter2;

      auto largest = std::max(axis1, axis2);
      *--output_shape = largest;

      if (largest == 1 && index + 1 < dimension_count_min)  // Nothing to do in this case
        continue;

      iterator1_.Init(axis1, largest);
      iterator2_.Init(axis2, largest);
      index++;  // Manually increment since we processed one axis
      break;
    }

    for (; index < dimension_count_min; index++) {
      auto axis1 = *--iter1;
      auto axis2 = *--iter2;

      auto largest = std::max(axis1, axis2);
      *--output_shape = largest;

      if (largest == 1)  // Nothing to do in this case
        continue;

      iterator1_.Append(axis1, largest);
      iterator2_.Append(axis2, largest);
    }

    // If one shape is bigger than another we need to broadcast the smaller onto the bigger from this point on
    for (; index < dimension_count_max; index++) {
      if (dimension_count_max == shape2.size()) {
        auto axis = *--iter2;
        iterator1_.Append(1, axis);
        iterator2_.Append(axis, axis);
        *--output_shape = axis;
      } else {
        auto axis = *--iter1;
        iterator1_.Append(axis, axis);
        iterator2_.Append(1, axis);
        *--output_shape = axis;
      }
    }

    // Allocate the counters
    iterator1_.counters_.resize(iterator1_.counts_.size(), 0);
    iterator2_.counters_.resize(iterator2_.counts_.size(), 0);
  }

  size_t GetSpanSize() const { return std::min(iterator1_.counts_.front(), iterator2_.counts_.front()); }

  BroadcastIterator iterator1_, iterator2_;
  std::vector<int64_t> output_shape_;
};

template <typename T, typename TOutput = T>
struct TBroadcaster {
  TBroadcaster(OpKernelContext& context)
      : input_tensor0_(*context.Input<Tensor>(0)),
        input_tensor1_(*context.Input<Tensor>(1)),
        output_tensor_(*context.Output(0, TensorShape(broadcaster_.output_shape_))) {
  }

  operator bool() const { return output_ != output_end_; }

  bool IsInput0Scalar() const { return broadcaster_.iterator1_.deltas_.front() == 0; }
  bool IsInput1Scalar() const { return broadcaster_.iterator2_.deltas_.front() == 0; }

  T NextScalar0() { return *Next0(); }
  T NextScalar1() { return *Next1(); }

  gsl::span<const T> NextSpan0() { return gsl::span<const T>(Next0(), span_size_); }
  gsl::span<const T> NextSpan1() { return gsl::span<const T>(Next1(), span_size_); }
  gsl::span<TOutput> NextSpanOutput() { return gsl::span<TOutput>(NextOutput(), span_size_); }

  ConstEigenVectorMap<T> NextEigen0() { return ConstEigenVectorMap<T>(Next0(), span_size_); }
  ConstEigenVectorMap<T> NextEigen1() { return ConstEigenVectorMap<T>(Next1(), span_size_); }
  EigenVectorMap<TOutput> NextEigenOutput() { return EigenVectorMap<TOutput>(NextOutput(), span_size_); }

 private:
  const T* Next0() { return input0_ + broadcaster_.iterator1_.AdvanceBy(span_size_); }
  const T* Next1() { return input1_ + broadcaster_.iterator2_.AdvanceBy(span_size_); }

  TOutput* NextOutput() {
    TOutput* output = output_;
    output_ += span_size_;
    return output;
  }

  const Tensor& input_tensor0_;
  const Tensor& input_tensor1_;
  Broadcaster broadcaster_{input_tensor0_.Shape().GetDims(), input_tensor1_.Shape().GetDims()};
  size_t span_size_{broadcaster_.GetSpanSize()};

  Tensor& output_tensor_;

  const T* input0_{input_tensor0_.Data<T>()};
  const T* input1_{input_tensor1_.Data<T>()};
  TOutput* output_{output_tensor_.MutableData<TOutput>()};
  const TOutput* output_end_{output_ + output_tensor_.Shape().Size()};
};

template <typename T, typename TOutput, typename Op>
void Loop(T scalar1, gsl::span<const T> input2, gsl::span<TOutput> output, Op op) {
  for (auto i = 0; i < output.size(); i++)
    output[i] = op(scalar1, input2[i]);
}

template <typename T, typename TOutput, typename Op>
void Loop(gsl::span<const T> input1, T scalar2, gsl::span<TOutput> output, Op op) {
  for (auto i = 0; i < output.size(); i++)
    output[i] = op(input1[i], scalar2);
}

template <>
Status Add<float>::Compute(OpKernelContext* context) const {
  TBroadcaster<float> bc(*context);

  if (bc.IsInput0Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextScalar0() + bc.NextEigen1().array();
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().array() + bc.NextScalar1();
  } else {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0() + bc.NextEigen1();
  }
  return Status::OK();
}

template <>
Status Sub<float>::Compute(OpKernelContext* context) const {
  TBroadcaster<float> bc(*context);

  if (bc.IsInput0Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextScalar0() - bc.NextEigen1().array();
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().array() - bc.NextScalar1();
  } else {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0() - bc.NextEigen1();
  }

  return Status::OK();
}

template <>
Status Mul<float>::Compute(OpKernelContext* context) const {
  TBroadcaster<float> bc(*context);

  if (bc.IsInput0Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextScalar0() * bc.NextEigen1().array();
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().array() * bc.NextScalar1();
  } else {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().cwiseProduct(bc.NextEigen1());
  }
  return Status::OK();
}

template <>
Status Div<float>::Compute(OpKernelContext* context) const {
  TBroadcaster<float> bc(*context);

  if (bc.IsInput0Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextScalar0() / bc.NextEigen1().array();
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().array() / bc.NextScalar1();
  } else {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().cwiseQuotient(bc.NextEigen1());
  }

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
Status Pow<float>::Compute(OpKernelContext* context) const {
  TBroadcaster<float> bc(*context);

  if (bc.IsInput0Scalar()) {
    while (bc) {
      float scalar = bc.NextScalar0();
      bc.NextEigenOutput() = EigenVectorMap<float>(&scalar, 1).array().pow(bc.NextEigen1().array());
    }
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().array().pow(bc.NextScalar1());
  } else {
    while (bc)
      bc.NextEigenOutput() = bc.NextEigen0().array().pow(bc.NextEigen1().array());
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
  auto input_count = Node().InputArgCount().front();
  LOTUS_ENFORCE(input_count >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto sum = EigenMap<float>(*ctx->Output(0, shape));

  if (input_count == 1) {
    sum = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    LOTUS_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    sum = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < input_count; index++) {
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

Status Not::Compute(OpKernelContext* context) const {
  auto& input = *context->Input<Tensor>(0);
  auto& output = *context->Output(0, input.Shape());

  EigenMap<bool>(output).array() = !EigenMap<bool>(input).array();
  return Status::OK();
}

Status And::Compute(OpKernelContext* context) const {
  auto op = [](bool a, bool b) { return a && b; };

  TBroadcaster<bool> bc(*context);
  if (bc.IsInput0Scalar()) {
    while (bc)
      Loop(bc.NextScalar0(), bc.NextSpan1(), bc.NextSpanOutput(), op);
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      Loop(bc.NextSpan0(), bc.NextScalar1(), bc.NextSpanOutput(), op);
  } else {
    while (bc)
      bc.NextEigenOutput().array() = bc.NextEigen0().array() && bc.NextEigen1().array();
  }
  return Status::OK();
}

Status Or::Compute(OpKernelContext* context) const {
  auto op = [](bool a, bool b) { return a || b; };

  TBroadcaster<bool> bc(*context);
  if (bc.IsInput0Scalar()) {
    while (bc)
      Loop(bc.NextScalar0(), bc.NextSpan1(), bc.NextSpanOutput(), op);
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      Loop(bc.NextSpan0(), bc.NextScalar1(), bc.NextSpanOutput(), op);
  } else {
    while (bc)
      bc.NextEigenOutput().array() = bc.NextEigen0().array() || bc.NextEigen1().array();
  }
  return Status::OK();
}

Status Xor::Compute(OpKernelContext* context) const {
  auto op = [](bool a, bool b) { return (a ^ b) != 0; };

  TBroadcaster<bool> bc(*context);
  if (bc.IsInput0Scalar()) {
    while (bc)
      Loop(bc.NextScalar0(), bc.NextSpan1(), bc.NextSpanOutput(), op);
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      Loop(bc.NextSpan0(), bc.NextScalar1(), bc.NextSpanOutput(), op);
  } else {
    while (bc)
      bc.NextEigenOutput().array() = bc.NextEigen0().array() ^ bc.NextEigen1().array();
  }
  return Status::OK();
}

template <>
Status Equal<int32_t>::Compute(OpKernelContext* context) const {
  auto op = [](auto a, auto b) { return a == b; };

  TBroadcaster<int32_t, bool> bc(*context);
  if (bc.IsInput0Scalar()) {
    while (bc)
      Loop(bc.NextScalar0(), bc.NextSpan1(), bc.NextSpanOutput(), op);
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      Loop(bc.NextSpan0(), bc.NextScalar1(), bc.NextSpanOutput(), op);
  } else {
    while (bc)
      bc.NextEigenOutput().array() = bc.NextEigen0().array() == bc.NextEigen1().array();
  }
  return Status::OK();
}

template <>
Status Less<float>::Compute(OpKernelContext* context) const {
  auto op = [](auto a, auto b) { return a < b; };

  TBroadcaster<float, bool> bc(*context);
  if (bc.IsInput0Scalar()) {
    while (bc)
      Loop(bc.NextScalar0(), bc.NextSpan1(), bc.NextSpanOutput(), op);
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      Loop(bc.NextSpan0(), bc.NextScalar1(), bc.NextSpanOutput(), op);
  } else {
    while (bc)
      bc.NextEigenOutput().array() = bc.NextEigen0().array() < bc.NextEigen1().array();
  }
  return Status::OK();
}

template <>
Status Greater<float>::Compute(OpKernelContext* context) const {
  auto op = [](auto a, auto b) { return a > b; };

  TBroadcaster<float, bool> bc(*context);
  if (bc.IsInput0Scalar()) {
    while (bc)
      Loop(bc.NextScalar0(), bc.NextSpan1(), bc.NextSpanOutput(), op);
  } else if (bc.IsInput1Scalar()) {
    while (bc)
      Loop(bc.NextSpan0(), bc.NextScalar1(), bc.NextSpanOutput(), op);
  } else {
    while (bc)
      bc.NextEigenOutput().array() = bc.NextEigen0().array() > bc.NextEigen1().array();
  }
  return Status::OK();
}

template <>
Status Mean<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  LOTUS_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto mean = EigenMap<float>(*ctx->Output(0, shape));

  if (inputCount == 1) {
    mean = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    LOTUS_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    mean = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < inputCount; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      LOTUS_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      mean += EigenMap<float>(data_n);
    }
  }

  // Take the mean
  float weight = 1.0f / static_cast<float>(inputCount);
  mean = mean * weight;

  return Status::OK();
}

template <>
Status Affine<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  MakeEigenArrayMap<float>(Y) = alpha_ * MakeEigenArrayMap<float>(X) + beta_;
  return Status::OK();
}

template <typename T>
class Sin final : public OpKernel {
 public:
  Sin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).sin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Sin,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sin<float>);

template <typename T>
class Cos final : public OpKernel {
 public:
  Cos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).cos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Cos,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cos<float>);

template <typename T>
class Tan final : public OpKernel {
 public:
  Tan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).tan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Tan,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Tan<float>);

template <typename T>
class Asin final : public OpKernel {
 public:
  Asin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).asin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Asin,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asin<float>);

template <typename T>
class Acos final : public OpKernel {
 public:
  Acos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).acos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Acos,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acos<float>);

template <typename T>
class Atan final : public OpKernel {
 public:
  Atan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).atan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Atan,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atan<float>);

template <>
Status PRelu<float>::Compute(OpKernelContext* context) const {
  TBroadcaster<float> bc(*context);

  if (bc.IsInput0Scalar()) {
    while (bc) {
      if (bc.NextScalar0() > 0)
        bc.NextEigenOutput().setConstant(bc.NextScalar0());
      else
        bc.NextEigenOutput() = bc.NextScalar0() * bc.NextEigen1().array();
    }
  } else if (bc.IsInput1Scalar()) {
    while (bc) {
      const auto& vec0 = bc.NextEigen0();
      bc.NextEigenOutput() = (vec0.array() > 0).select(vec0, vec0 * bc.NextScalar1());
    }
  } else {
    while (bc) {
      const auto& vec0 = bc.NextEigen0();
      bc.NextEigenOutput() = (vec0.array() > 0).select(vec0, vec0.cwiseProduct(bc.NextEigen1()));
    }
  }
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    PRelu,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

}  // namespace Lotus
