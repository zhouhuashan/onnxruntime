#include "core/providers/cpu/ml/svmregressor.h"

namespace Lotus {
namespace ML {

REGISTER_KERNEL(KernelDefBuilder("SVMRegressor")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                SVMRegressor<float>);

template <typename T>
SVMRegressor<T>::SVMRegressor(const OpKernelInfo& info) : OpKernel(info), SVMCommon<T>(info) {
  vector_count_ = 0;
  op_kernel_info_.GetAttr<int64_t>("n_supports", &vector_count_);

  LOTUS_ENFORCE(op_kernel_info_.GetAttrs<float>("rho", rho_).IsOK());
  LOTUS_ENFORCE(op_kernel_info_.GetAttrs<float>("coefficients", coefficients_).IsOK());
  LOTUS_ENFORCE(coefficients_.size() > 0);

  // optional for linear
  op_kernel_info_.GetAttrs<float>("support_vectors", support_vectors_);

  int64_t onec = 0;
  op_kernel_info_.GetAttr<int64_t>("one_class", &onec);
  one_class_ = (onec != 0);

  std::string tmp = "NONE";
  op_kernel_info_.GetAttr<std::string>("post_transform", &tmp);
  POST_EVAL_TRANSFORM tmpval = MakeTransform(tmp);
  post_transform_ = tmpval;

  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size();
    mode_ = SVM_TYPE::SVM_LINEAR;
    set_kernel_type(KERNEL::LINEAR);
  }
}

template <typename T>
Status SVMRegressor<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  int64_t stride = X->Shape().Size() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().Size() == 1 ? 1 : X->Shape()[0];

  Tensor* Y = ctx->Output(0, TensorShape({N, 1}));  // this op outputs for one target only
  const auto* x_data = X->Data<T>();

  for (int64_t n = 0; n < N; n++) {  //for each example
    int64_t current_weight_0 = n * stride;
    std::vector<float> scores;

    float sum = 0.f;
    if (mode_ == SVM_TYPE::SVM_SVC) {
      for (int64_t j = 0; j < vector_count_; j++) {
        float val1 = kernel_dot(x_data, current_weight_0, support_vectors_, feature_count_ * j, feature_count_, get_kernel_type());
        float val2 = coefficients_[j];
        float val3 = val1 * val2;
        sum += val3;
      }
      sum += rho_[0];
    } else if (mode_ == SVM_TYPE::SVM_LINEAR) {  //liblinear
      sum = kernel_dot(x_data, current_weight_0, coefficients_, 0, feature_count_, get_kernel_type());
      sum += rho_[0];
    }
    if (one_class_ && sum > 0) {
      Y->MutableData<float>()[n] = 1.f;
    } else if (one_class_) {
      Y->MutableData<float>()[n] = -1.f;
    } else {
      Y->MutableData<float>()[n] = sum;
    }
  }

  return Status::OK();
}

}  // namespace ML
}  // namespace Lotus
