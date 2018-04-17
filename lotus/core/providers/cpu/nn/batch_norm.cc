#include "core/providers/cpu/nn/batch_norm.h"

namespace Lotus {
// spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
REGISTER_KERNEL(KernelDefBuilder("BatchNormalization")
                    .Domain(LotusIR::kOnnxDomain)
                    // This operator is used if you are using version 6 of the default ONNX operator
                    // set until the next BC-breaking change to this operator
                    .SinceVersion(6)
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
  if (X->Shape().GetDims().empty()) {
    std::ostringstream ostr;
    ostr << "Invalid input X: Empty dimensions";
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (scale->Shape().NumDimensions() != kNumInputScaleDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input scale: NumDimensions() != " << kNumInputScaleDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (B->Shape().NumDimensions() != kNumInputBiasDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input B: NumDimensions() != " << kNumInputBiasDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (mean->Shape().NumDimensions() != kNumInputMeanDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input mean: NumDimensions() != " << kNumInputMeanDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }
  if (var->Shape().NumDimensions() != kNumInputVarianceDimensions) {
    std::ostringstream ostr;
    ostr << "Invalid input var: NumDimensions() != " << kNumInputVarianceDimensions;
    return Status(LOTUS, INVALID_ARGUMENT, ostr.str());
  }

  return Status::OK();
}

template <>
Status BatchNorm<float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* var = p_op_kernel_context->Input<Tensor>(4);

  LOTUS_RETURN_IF_ERROR(ValidateInputs(X, scale, B, mean, var));

  const TensorShape& x_shape = X->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  const auto& dims_vec = x_shape.GetDims();
  const size_t N = dims_vec[0];
  const size_t C = dims_vec[1];  // assume NCHW as per the spec

  // calculate sample_size
  size_t sample_size = 1;
  for (size_t i = 2; i < dims_vec.size(); ++i) {
    sample_size *= dims_vec[i];
  }

  ConstEigenVectorArrayMap<float> scale_arr(scale->Data<float>(), C);
  ConstEigenVectorArrayMap<float> bias_arr(B->Data<float>(), C);

  // Regardless of training or testing, we will apply the estimated mean
  // and standard deviation to the input. For testing, they are
  // specified directly by the input, and for training, they are computed
  // by the op.
  Eigen::Array<float, Eigen::Dynamic, 1> inv_std(C);
  ConstEigenVectorArrayMap<float> var_arr(var->Data<float>(), C);
  inv_std = (var_arr + epsilon_).sqrt().inverse();
  ConstEigenVectorArrayMap<float> mean_arr(mean->Data<float>(), C);
  // We can fuse the output computation as follows:
  //   ((x - est_mean) * (inv_var) * scale + bias
  // to
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  Eigen::Array<float, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<float, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
  EigenArrayMap<float> Y_arr(Y->MutableData<float>(), sample_size, N * C);
  ConstEigenArrayMap<float> X_arr(X->Data<float>(), sample_size, N * C);
  for (int nc = 0; nc < N * C; ++nc) {
    Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
  }

  return Status::OK();
}
}  // namespace Lotus
