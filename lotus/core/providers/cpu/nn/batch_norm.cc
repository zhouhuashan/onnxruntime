#include "core/providers/cpu/nn/batch_norm.h"

namespace Lotus {
// spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
REGISTER_KERNEL(KernelDefBuilder("BatchNormalization")
                    .Domain(LotusIR::kOnnxDomain)
                    // This operator is used if you are using version 6 of the default ONNX operator
                    // set until the next BC-breaking change to this operator
                    .SinceVersion(6, 7)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("X", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("scale", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("mean", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("var", DataTypeImpl::GetTensorType<float>()),
                BatchNorm<float>);

template <>
Status BatchNorm<float>::ValidateInputs(const Tensor* X,
                                        const Tensor* scale,
                                        const Tensor* B,
                                        const Tensor* mean,
                                        const Tensor* var) const {
  if (X->shape().NumDimensions() != kNumInputXDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input X: NumDimensions() != " << kNumInputXDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (scale->shape().NumDimensions() != kNumInputScaleDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input scale: NumDimensions() != " << kNumInputScaleDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (B->shape().NumDimensions() != kNumInputBiasDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input B: NumDimensions() != " << kNumInputBiasDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (mean->shape().NumDimensions() != kNumInputMeanDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input mean: NumDimensions() != " << kNumInputMeanDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (var->shape().NumDimensions() != kNumInputVarianceDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input var: NumDimensions() != " << kNumInputVarianceDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }

  return Status::OK();
}

template <>
Status BatchNorm<float>::compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* X = p_op_kernel_context->input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->input<Tensor>(3);
  const Tensor* var = p_op_kernel_context->input<Tensor>(4);

  LOTUS_RETURN_IF_ERROR(ValidateInputs(X, scale, B, mean, var));

  const TensorShape& x_shape = X->shape();
  Tensor* Y = p_op_kernel_context->output(0, x_shape);

  const size_t N = x_shape[0];
  const size_t C = x_shape[1];  // assume NCHW as per the spec
  const size_t H = x_shape[2];
  const size_t W = x_shape[3];

  const size_t sample_size = H * W;

  ConstEigenVectorArrayMap<float> scale_arr(scale->data<float>(), C);
  ConstEigenVectorArrayMap<float> bias_arr(B->data<float>(), C);

  // Regardless of training or testing, we will apply the estimated mean
  // and standard deviation to the input. For testing, they are
  // specified directly by the input, and for training, they are computed
  // by the op.
  Eigen::Array<float, Eigen::Dynamic, 1> inv_std(C);
  ConstEigenVectorArrayMap<float> var_arr(var->data<float>(), C);
  inv_std = (var_arr + epsilon_).sqrt().inverse();
  ConstEigenVectorArrayMap<float> mean_arr(mean->data<float>(), C);
  // We can fuse the output computation as follows:
  //   ((x - est_mean) * (inv_var) * scale + bias
  // to
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  Eigen::Array<float, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<float, Eigen::Dynamic, 1> new_bias =
      bias_arr - mean_arr * inv_std * scale_arr;
  EigenArrayMap<float> Y_arr(Y->mutable_data<float>(), sample_size, N * C);
  ConstEigenArrayMap<float> X_arr(X->data<float>(), sample_size, N * C);
  for (int nc = 0; nc < N * C; ++nc) {
    Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
  }

  return Status::OK();
}
}  // namespace Lotus
