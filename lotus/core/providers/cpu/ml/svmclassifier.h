#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "ml_common.h"

namespace Lotus {
namespace ML {

template <typename T>
class SVMClassifier final : public OpKernel {
 public:
  SVMClassifier(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  float kernel_dot(const T* A, int64_t a, const std::vector<float>& B, int64_t b, int64_t len, KERNEL k) const;

 private:
  bool weights_are_all_positive_;
  KERNEL kernel_type_;
  float gamma_;
  float coef0_;
  float degree_;
  int64_t feature_count_;
  int64_t class_count_;
  int64_t vector_count_;
  bool using_strings_;
  std::vector<int64_t> vectors_per_class_;
  std::vector<int64_t> starting_vector_;
  std::vector<float> rho_;
  std::vector<float> proba_;
  std::vector<float> probb_;
  std::vector<float> coefficients_;
  std::vector<float> support_vectors_;
  std::vector<int64_t> classlabels_ints_;
  std::vector<std::string> classlabels_strings_;
  POST_EVAL_TRANSFORM post_transform_;
  SVM_TYPE mode_;  //how are we computing SVM? 0=LibSVC, 1=LibLinear
};
}  // namespace ML
}  // namespace Lotus
