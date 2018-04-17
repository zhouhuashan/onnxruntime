#include "core/providers/cpu/ml/svmclassifier.h"

namespace Lotus {
namespace ML {

REGISTER_KERNEL(KernelDefBuilder("SVMClassifier")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}),
                SVMClassifier<float>);

template <typename T>
float SVMClassifier<T>::kernel_dot(const T* A, int64_t a, const std::vector<float>& B, int64_t b, int64_t len, KERNEL k) const {
  float sum = 0.f;

  if (k == KERNEL::POLY) {
    for (int64_t i = 0; i < len; i++) {
      sum += B[b + i] * static_cast<float>(A[a + i]);
    }
    sum = gamma_ * sum + coef0_;
    sum = std::pow(sum, degree_);
  } else if (k == KERNEL::SIGMOID) {
    for (int64_t i = 0; i < len; i++) {
      sum += B[b + i] * static_cast<float>(A[a + i]);
    }
    sum = gamma_ * sum + coef0_;
    sum = std::tanh(sum);
  } else if (k == KERNEL::RBF) {
    for (int64_t i = 0; i < len; i++) {
      float val = static_cast<float>(A[a + i]) - B[b + i];
      sum += (val * val);
    }
    sum = std::exp(-gamma_ * sum);
  } else if (k == KERNEL::LINEAR) {
    for (int64_t i = 0; i < len; i++) {
      sum += B[b + i] * static_cast<float>(A[a + i]);
    }
  }
  return sum;
}

template <typename T>
SVMClassifier<T>::SVMClassifier(const OpKernelInfo& info) : OpKernel(info) {
  LOTUS_ENFORCE(op_kernel_info_.GetAttrs<float>("rho", rho_).IsOK());
  LOTUS_ENFORCE(op_kernel_info_.GetAttrs<float>("coefficients", coefficients_).IsOK());

  // optional attributes
  op_kernel_info_.GetAttrs<int64_t>("vectors_per_class", vectors_per_class_);
  op_kernel_info_.GetAttrs<float>("support_vectors", support_vectors_);

  // prob_a and prob_b are optional for Z output
  op_kernel_info_.GetAttrs<float>("prob_a", proba_);
  op_kernel_info_.GetAttrs<float>("prob_b", probb_);
  LOTUS_ENFORCE(proba_.size() == probb_.size());

  // one of these should be valid
  LOTUS_ENFORCE(op_kernel_info_.GetAttrs<std::string>("classlabels_strings", classlabels_strings_).IsOK() ||
                op_kernel_info_.GetAttrs<int64_t>("classlabels_ints", classlabels_ints_).IsOK());

  std::vector<float> kernel_params;
  LOTUS_ENFORCE(op_kernel_info_.GetAttrs<float>("kernel_params", kernel_params).IsOK());

  std::string tmp = "NONE";
  op_kernel_info_.GetAttr<std::string>("post_transform", &tmp);
  POST_EVAL_TRANSFORM tmpval = MakeTransform(tmp);
  post_transform_ = tmpval;

  tmp = "LINEAR";
  op_kernel_info_.GetAttr<std::string>("kernel_type", &tmp);
  KERNEL tmpval2 = MakeKernel(tmp);
  kernel_type_ = tmpval2;

  if (kernel_params.size() > 0) {
    gamma_ = kernel_params[0];
    coef0_ = kernel_params[1];
    degree_ = kernel_params[2];
  }

  vector_count_ = 0;
  feature_count_ = 0;
  class_count_ = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(vectors_per_class_.size()); i++) {
    starting_vector_.push_back(vector_count_);
    vector_count_ += vectors_per_class_[i];
  }

  using_strings_ = false;
  if (classlabels_strings_.size() > 0) {
    using_strings_ = true;
    class_count_ = classlabels_strings_.size();
  } else if (classlabels_ints_.size() > 0) {
    class_count_ = classlabels_ints_.size();
  } else {
    class_count_ = 1;
  }
  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size() / class_count_;  //liblinear mode
    mode_ = SVM_TYPE::SVM_LINEAR;
    kernel_type_ = KERNEL::LINEAR;
  }
  LOTUS_ENFORCE(classlabels_strings_.size() > 0 || classlabels_ints_.size() > 0);
  LOTUS_ENFORCE(proba_.size() == probb_.size());
  LOTUS_ENFORCE(coefficients_.size() > 0);
  weights_are_all_positive_ = true;
  for (int64_t i = 0; i < static_cast<int64_t>(coefficients_.size()); i++) {
    if (coefficients_[i] < 0) {
      weights_are_all_positive_ = false;
      break;
    }
  }
}

template <typename T>
Status SVMClassifier<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  int64_t stride = X->Shape().Size() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().Size() == 1 ? 1 : X->Shape()[0];

  Tensor* Y = ctx->Output(0, TensorShape({N}));
  Tensor* Z;

  std::vector<int64_t> dims;
  if (mode_ == SVM_TYPE::SVM_SVC && proba_.size() == 0)
    dims = {static_cast<int64_t>(N), static_cast<int64_t>(class_count_ * (class_count_ - 1) / 2)};
  else
    dims = {static_cast<int64_t>(N), static_cast<int64_t>(class_count_)};
  Z = ctx->Output(1, TensorShape(dims));

  const auto* x_data = X->Data<T>();
  int64_t zindex = 0;

  for (int64_t n = 0; n < N; n++)  //for each example
  {
    int64_t current_weight_0 = n * stride;
    int64_t maxclass = -1;
    double maxweight = 0.f;
    std::vector<float> decisions;
    std::vector<float> scores;
    std::vector<float> kernels;
    std::vector<int64_t> votes;

    if (mode_ == SVM_TYPE::SVM_SVC) {
      for (int64_t j = 0; j < vector_count_; j++) {
        float val = kernel_dot(x_data, current_weight_0, support_vectors_, feature_count_ * j, feature_count_, kernel_type_);
        kernels.push_back(val);
      }
      for (int64_t j = 0; j < class_count_; j++) {
        votes.push_back(0);
      }
      int evals = 0;
      for (int64_t i = 0; i < class_count_; i++) {        //for each class
        for (int64_t j = i + 1; j < class_count_; j++) {  //for each class
          float sum = 0;
          int64_t start_index_i = starting_vector_[i];  // *feature_count_;
          int64_t start_index_j = starting_vector_[j];  // *feature_count_;

          int64_t class_i_support_count = vectors_per_class_[i];
          int64_t class_j_support_count = vectors_per_class_[j];

          int64_t pos1 = (vector_count_) * (j - 1);
          int64_t pos2 = (vector_count_) * (i);
          for (int64_t m = 0; m < class_i_support_count; m++) {
            float val1 = coefficients_[pos1 + start_index_i + m];
            float val2 = kernels[start_index_i + m];
            sum += val1 * val2;
          }
          for (int64_t m = 0; m < class_j_support_count; m++) {
            float val1 = coefficients_[pos2 + start_index_j + m];
            float val2 = kernels[start_index_j + m];
            sum += val1 * val2;
          }

          sum += rho_[evals];
          scores.push_back(sum);
          if (sum > 0) {
            votes[i]++;
          } else {
            votes[j]++;
          }
          evals++;  //index into rho
        }
      }
    } else if (mode_ == SVM_TYPE::SVM_LINEAR) {     //liblinear
      for (int64_t j = 0; j < class_count_; j++) {  //for each class
        float val = kernel_dot(x_data, current_weight_0, coefficients_, feature_count_ * j, feature_count_, kernel_type_);
        val += rho_[0];
        scores.push_back(val);
      }
    }
    if (proba_.size() > 0 && mode_ == SVM_TYPE::SVM_SVC) {
      //compute probabilities from the scores
      std::vector<float> estimates;
      std::vector<float> probsp2;
      int64_t num = class_count_ * class_count_;
      for (int64_t m = 0; m < num; m++) {
        probsp2.push_back(0.f);  //min prob
      }
      for (int64_t m = 0; m < class_count_; m++) {
        estimates.push_back(0.f);  //min prob
      }
      int64_t index = 0;
      for (int64_t i = 0; i < class_count_; i++) {
        for (int64_t j = i + 1; j < class_count_; j++) {
          float val1 = sigmoid_probability(scores[index], proba_[index], probb_[index]);
          float val2 = std::max(val1, 1.0e-7f);
          probsp2[i * class_count_ + j] = std::min(val2, 1 - 1.0e-7f);
          probsp2[j * class_count_ + i] = 1 - probsp2[i * class_count_ + j];
          index++;
        }
      }
      multiclass_probability(class_count_, probsp2, estimates);
      //copy probabilities back into scores
      scores.resize(estimates.size());
      for (int64_t k = 0; k < static_cast<int64_t>(estimates.size()); k++) {
        scores[k] = estimates[k];
      }
    }
    int64_t maxvotes = 0;
    if (votes.size() > 0) {
      for (int64_t k = 0; k < static_cast<int64_t>(votes.size()); k++) {
        if (votes[k] > maxvotes) {
          maxvotes = votes[k];
          maxclass = k;
        }
      }
    } else {
      for (int64_t k = 0; k < static_cast<int64_t>(scores.size()); k++) {
        if (scores[k] > maxweight) {
          maxclass = k;
          maxweight = scores[k];
        }
      }
    }
    //write top class
    int write_additional_scores = -1;
    if (rho_.size() == 1)  //binary
    {
      if (using_strings_) {
        if (classlabels_strings_.size() == 2 && weights_are_all_positive_ && maxweight >= 0.5 && proba_.size() == 0) {
          Y->MutableData<std::string>()[n] = classlabels_strings_[1];  //positive label
          write_additional_scores = 0;
        } else if (classlabels_strings_.size() == 2 && maxweight > 0 && !weights_are_all_positive_ && proba_.size() == 0) {
          Y->MutableData<std::string>()[n] = classlabels_strings_[1];  //positive label
          write_additional_scores = 0;
        } else if (classlabels_strings_.size() == 2 && proba_.size() > 0) {   //this case all classes are in their rightful spot
          Y->MutableData<std::string>()[n] = classlabels_strings_[maxclass];  //whichever label
          write_additional_scores = -1;
        } else if (classlabels_strings_.size() == 2) {
          Y->MutableData<std::string>()[n] = classlabels_strings_[0];  //negative label
          write_additional_scores = 1;
        } else if (maxweight > 0) {
          Y->MutableData<int64_t>()[n] = 1;  //positive label
        } else {
          Y->MutableData<int64_t>()[n] = 0;  //negative label
        }
      } else  //no strings
      {
        if (classlabels_ints_.size() == 2 && weights_are_all_positive_ && maxweight >= 0.5 && proba_.size() == 0) {
          Y->MutableData<int64_t>()[n] = classlabels_ints_[1];  //positive label
          write_additional_scores = 0;
        } else if (classlabels_ints_.size() == 2 && maxweight > 0 && !weights_are_all_positive_ && proba_.size() == 0) {
          Y->MutableData<int64_t>()[n] = classlabels_ints_[0];  //pos  label
          write_additional_scores = 0;
        } else if (classlabels_ints_.size() == 2 && proba_.size() > 0)  //this case all classes are in their rightful spot
        {
          Y->MutableData<int64_t>()[n] = classlabels_ints_[maxclass];  //whichever label
          write_additional_scores = -1;
        } else if (classlabels_ints_.size() == 2) {
          Y->MutableData<int64_t>()[n] = classlabels_ints_[0];  //negative label
          write_additional_scores = 1;
        } else if (maxweight > 0) {
          Y->MutableData<int64_t>()[n] = 1;  //positive label
        } else {
          Y->MutableData<int64_t>()[n] = 0;  //negative label
        }
      }
    } else {  //multiclass
      if (using_strings_) {
        Y->MutableData<std::string>()[n] = classlabels_strings_[maxclass];
      } else {
        Y->MutableData<int64_t>()[n] = classlabels_ints_[maxclass];
      }
    }

    write_scores(scores, post_transform_, zindex, Z, write_additional_scores);
    zindex += scores.size();
  }

  return Status::OK();
}

}  // namespace ML
}  // namespace Lotus
