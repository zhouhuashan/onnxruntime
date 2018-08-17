#include "core/providers/cpu/nn/instance_norm.h"
#include "core/util/math_cpuonly.h"
using namespace ::Lotus::Common;

namespace Lotus {

ONNX_CPU_OPERATOR_KERNEL(
    InstanceNormalization,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    InstanceNorm<float>);

static Status ValidateInputs(const Tensor* input, const Tensor* scale, const Tensor* B) {
  if (input->Shape().NumDimensions() < 3) {
    std::ostringstream ostr;
    ostr << "Invalid input data: number of dimensions is less than 3: " << input->Shape().NumDimensions();
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (scale->Shape().NumDimensions() != 1) {
    std::ostringstream ostr;
    ostr << "Invalid input scale: number of dimensions is not 1: " << scale->Shape().NumDimensions();
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (scale->Shape().Size() != input->Shape().GetDims()[1]) {
    std::ostringstream ostr;
    ostr << "Mismatch between input data and scale: size of scale != input channel count "
         << scale->Shape().Size() << " vs. " << input->Shape().GetDims()[1];
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }

  if (B->Shape().NumDimensions() != 1) {
    std::ostringstream ostr;
    ostr << "Invalid input B: number of dimensions is not 1: " << B->Shape().NumDimensions();
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }

  if (B->Shape().Size() != input->Shape().GetDims()[1]) {
    std::ostringstream ostr;
    ostr << "Mismatch between input data and B: size of B != input channel count "
         << B->Shape().Size() << " vs. " << input->Shape().GetDims()[1];
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }

  return Status::OK();
}

template <>
Status InstanceNorm<float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* input = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->Input<Tensor>(2);

  LOTUS_RETURN_IF_ERROR(ValidateInputs(input, scale, B));
  const int64_t N = input->Shape().GetDims()[0];
  const int64_t C = input->Shape().GetDims()[1];
  const int64_t W = input->Shape().SizeFromDimension(2);

  const TensorShape& x_shape = input->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  for (auto i = 0; i < N * C; ++i) {
    ConstEigenVectorArrayMap<float> Xi(input->Data<float>() + W * i, W);
    const float Xi_mean = Xi.mean();
    const float squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const float inv_stdev = 1.0f / std::sqrt(squared_norm / W + epsilon_);
    EigenVectorArrayMap<float> Yi(Y->MutableData<float>() + W * i, W);
    const float channel_scale = inv_stdev * scale->Data<float>()[i % C];
    const float channel_shift = B->Data<float>()[i % C] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
  }

  return Status::OK();
}
}  // namespace Lotus
