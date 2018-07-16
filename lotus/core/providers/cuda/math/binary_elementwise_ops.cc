#include "binary_elementwise_ops.h"
#include "core/providers/cpu/tensor/utils.h"
#include "binary_elementwise_ops_impl.h"
using namespace Lotus::Common;
namespace Lotus {
namespace Cuda {

template <>
Status BinaryElementwise<ShouldNotBroadcast>::Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const {
  p->lhs_tensor = context->Input<Tensor>(0);
  p->rhs_tensor = context->Input<Tensor>(1);
  if (!(p->lhs_tensor->Shape() == p->rhs_tensor->Shape()))
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, Node().Name(), ": mismatching input shapes: ", p->lhs_tensor->Shape().ToString(), " != ", p->rhs_tensor->Shape().ToString());
  p->output_tensor = context->Output(0, p->lhs_tensor->Shape());
  p->output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
  return Status::OK();
}

static Status ComputeOutputShape(const std::string& node_name, const TensorShape& lhs_shape, const TensorShape& rhs_shape, TensorShape& out_shape) {
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (int i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank)
      lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank)
      rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t out_dim = std::max(lhs_dim, rhs_dim);
    if (lhs_dim != out_dim && lhs_dim != 1)
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, node_name, ": left operand cannot broadcast on dim ", lhs_rank - 1 - i,
                               " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    if (rhs_dim != out_dim && rhs_dim != 1)
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, node_name, ": right operand cannot broadcast on dim ", rhs_rank - 1 - i,
                               " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = TensorShape(output_dims);
  return Status::OK();
}

Status BinaryElementwiseBroadcastPrepare(const Tensor* lhs_tensor, const Tensor* rhs_tensor, Tensor* output_tensor, BinaryElementwisePreparation* p) {
  p->lhs_tensor = lhs_tensor;
  p->rhs_tensor = rhs_tensor;
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  p->output_tensor = output_tensor;

  // early return when shapes match
  if (lhs_shape == rhs_shape) {
    p->output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
    return Status::OK();
  }

  // early return if one operand is scalar
  if (lhs_shape.Size() <= 1 || rhs_shape.Size() <= 1) {
    p->output_rank_or_simple_broadcast = static_cast<size_t>(lhs_shape.Size() <= 1 ? SimpleBroadcast::LeftScalar : SimpleBroadcast::RightScalar);
    return Status::OK();
  }

  p->output_rank_or_simple_broadcast = out_rank;

  p->lhs_padded_strides.AllocCpuPtr(out_rank);
  p->rhs_padded_strides.AllocCpuPtr(out_rank);
  p->fdm_output_strides.AllocCpuPtr(out_rank);
  LOTUS_ENFORCE(TensorPitches::Calculate(p->lhs_padded_strides.CpuSpan(), p->lhs_tensor->Shape().GetDims()));
  LOTUS_ENFORCE(TensorPitches::Calculate(p->rhs_padded_strides.CpuSpan(), p->rhs_tensor->Shape().GetDims()));
  LOTUS_ENFORCE(CalculateFdmStrides(p->fdm_output_strides.CpuSpan(), output_tensor->Shape().GetDims()));

  p->lhs_dim0_broadcast = (lhs_rank == 0 || lhs_shape[0] == 1 || lhs_rank < out_rank);
  p->rhs_dim0_broadcast = (rhs_rank == 0 || rhs_shape[0] == 1 || rhs_rank < out_rank);

  LOTUS_RETURN_IF_ERROR(p->lhs_padded_strides.CopyToGpu());
  LOTUS_RETURN_IF_ERROR(p->rhs_padded_strides.CopyToGpu());
  LOTUS_RETURN_IF_ERROR(p->fdm_output_strides.CopyToGpu());
  return Status::OK();
}

template <>
Status BinaryElementwise<ShouldBroadcast>::Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const {
  auto lhs_tensor = context->Input<Tensor>(0);
  auto rhs_tensor = context->Input<Tensor>(1);
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();

  TensorShape output_shape;
  LOTUS_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), lhs_shape, rhs_shape, output_shape));
  auto output_tensor = context->Output(0, output_shape);

  LOTUS_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(lhs_tensor, rhs_tensor, output_tensor, p));

  return Status::OK();
}

#define BINARY_ELEMENTWISE_REGISTER_KERNEL(x, ver, T)                         \
  REGISTER_KERNEL(KernelDefBuilder(#x)                                        \
                      .Domain(LotusIR::kOnnxDomain)                           \
                      .SinceVersion(ver)                                      \
                      .Provider(LotusIR::kCudaExecutionProvider)              \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                  x<T>);

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                \
  template <>                                                                                           \
  Status x<T>::Compute(OpKernelContext* context) const {                                                \
    BinaryElementwisePreparation prepare(provider_);                                                    \
    Prepare(context, &prepare);                                                                         \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                       \
        prepare.output_rank_or_simple_broadcast,                                                        \
        prepare.lhs_dim0_broadcast,                                                                     \
        prepare.lhs_padded_strides.GpuPtr(),                                                            \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->Data<T>()),     \
        prepare.rhs_dim0_broadcast,                                                                     \
        prepare.rhs_padded_strides.GpuPtr(),                                                            \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->Data<T>()),     \
        prepare.fdm_output_strides.GpuPtr(),                                                            \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->MutableData<T>()), \
        prepare.output_tensor->Shape().Size());                                                         \
                                                                                                        \
    return Status::OK();                                                                                \
  }

#define BINARY_OP_TYPED(name, ver, T)              \
  BINARY_ELEMENTWISE_REGISTER_KERNEL(name, ver, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

// since different ops has different types, we cannot use BINARY_OPS() directly
// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#define BINARY_OP_HFD(name, ver)        \
  BINARY_OP_TYPED(name, ver, MLFloat16) \
  BINARY_OP_TYPED(name, ver, float)     \
  BINARY_OP_TYPED(name, ver, double)

#define BINARY_OP_UZILHFD(name, ver)   \
  BINARY_OP_TYPED(name, ver, uint32_t) \
  BINARY_OP_TYPED(name, ver, uint64_t) \
  BINARY_OP_TYPED(name, ver, int32_t)  \
  BINARY_OP_TYPED(name, ver, int64_t)  \
  BINARY_OP_HFD(name, ver)

BINARY_OP_UZILHFD(Add, 7)
BINARY_OP_UZILHFD(Sub, 7)
BINARY_OP_UZILHFD(Mul, 7)
BINARY_OP_UZILHFD(Div, 7)
BINARY_OP_HFD(Pow, 7)
BINARY_OP_TYPED(And, 7, bool)
BINARY_OP_TYPED(Or, 7, bool)
BINARY_OP_TYPED(Xor, 7, bool)
BINARY_OP_HFD(PRelu, 7)

template <typename T>
Status Sum<T>::Compute(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto& node = Node();
  const auto& node_name = node.Name();
  auto input_count = node.InputArgCount().front();
  LOTUS_ENFORCE(input_count >= 1, "Must have 1 or more inputs");

  if (input_count == 1) {
    auto input_tensor = context->Input<Tensor>(0);
    const auto& input_shape = input_tensor->Shape();
    auto output_tensor = context->Output(0, input_shape);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(output_tensor->MutableDataRaw(), input_tensor->DataRaw(), sizeof(CudaT) * input_shape.Size(), cudaMemcpyDeviceToDevice));
  } else {
    // compute output shape first, using broadcast rule
    TensorShape output_shape;
    LOTUS_RETURN_IF_ERROR(ComputeOutputShape(node_name, context->Input<Tensor>(0)->Shape(), context->Input<Tensor>(1)->Shape(), output_shape));
    for (int index = 2; index < input_count; index++) {
      TensorShape previous_output_shape = output_shape;
      LOTUS_RETURN_IF_ERROR(ComputeOutputShape(node_name, previous_output_shape, context->Input<Tensor>(index)->Shape(), output_shape));
    }
    Tensor* output_tensor = context->Output(0, output_shape);
    BinaryElementwisePreparation prepare(provider_);
    if (input_count == 2) {
      // special case for 2 tensors to avoid memset zero
      LOTUS_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(context->Input<Tensor>(0), context->Input<Tensor>(1), output_tensor, &prepare));
      Impl_Add<CudaT>(
          prepare.output_rank_or_simple_broadcast,
          prepare.lhs_dim0_broadcast,
          prepare.lhs_padded_strides.GpuPtr(),
          reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()),
          prepare.rhs_dim0_broadcast,
          prepare.rhs_padded_strides.GpuPtr(),
          reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()),
          prepare.fdm_output_strides.GpuPtr(),
          reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
          prepare.output_tensor->Shape().Size());
    } else {
      // for more than 2 inputs, we need to accumulate into output tensor, as the shape from input0 + input1 might be different from output shape
      CUDA_RETURN_IF_ERROR(cudaMemset(output_tensor->MutableDataRaw(), 0, output_shape.Size() * sizeof(CudaT)));
      for (int index = 0; index < input_count; index++) {
        LOTUS_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(output_tensor, context->Input<Tensor>(index), output_tensor, &prepare));
        Impl_Add<CudaT>(
            prepare.output_rank_or_simple_broadcast,
            prepare.lhs_dim0_broadcast,
            prepare.lhs_padded_strides.GpuPtr(),
            reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()),
            prepare.rhs_dim0_broadcast,
            prepare.rhs_padded_strides.GpuPtr(),
            reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()),
            prepare.fdm_output_strides.GpuPtr(),
            reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
            prepare.output_tensor->Shape().Size());
      }
    }
  }

  return Status::OK();
}

#undef BINARY_ELEMENTWISE_COMPUTE
#define BINARY_ELEMENTWISE_COMPUTE(name, T) \
  template Status name<T>::Compute(OpKernelContext* context) const;

BINARY_OP_UZILHFD(Sum, 6)  // bump up this when upgrading ONNX to current
}  // namespace Cuda
}  // namespace Lotus
