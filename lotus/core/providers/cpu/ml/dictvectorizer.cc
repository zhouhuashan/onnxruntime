#include "core/providers/cpu/ml/dictvectorizer.h"

namespace Lotus {
namespace ML {
Status DictVectorizerOp::Compute(OpKernelContext* context) const {
  auto input_type = context->InputType(0);
  if (input_type == DataTypeImpl::GetType<std::map<std::string, int64_t> >()) {
    return ComputeWithType<std::string, int64_t>(context, string_index_, 0);
  } else if (input_type == DataTypeImpl::GetType<std::map<int64_t, std::string> >()) {
    return ComputeWithType<int64_t, std::string>(context, int_index_, "");
  } else if (input_type == DataTypeImpl::GetType<std::map<std::string, float> >()) {
    return ComputeWithType<std::string, float>(context, string_index_, 0.f);
  } else if (input_type == DataTypeImpl::GetType<std::map<int64_t, float> >()) {
    return ComputeWithType<int64_t, float>(context, int_index_, 0.f);
  } else if (input_type == DataTypeImpl::GetType<std::map<int64_t, double> >()) {
    return ComputeWithType<int64_t, double>(context, int_index_, 0.f);
  } else if (input_type == DataTypeImpl::GetType<std::map<std::string, double> >()) {
    return ComputeWithType<std::string, double>(context, string_index_, 0.f);
  } else {
    return Status(LOTUS, INVALID_ARGUMENT, "DictVectorizer: input has wrong type");
  }
}

REGISTER_KERNEL(KernelDefBuilder("DictVectorizer")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T1", std::vector<MLDataType>{
                                              DataTypeImpl::GetType<std::map<std::string, int64_t> >(),
                                              DataTypeImpl::GetType<std::map<int64_t, std::string> >(),
                                              DataTypeImpl::GetType<std::map<int64_t, float> >(),
                                              DataTypeImpl::GetType<std::map<int64_t, double> >(),
                                              DataTypeImpl::GetType<std::map<std::string, double> >(),
                                              DataTypeImpl::GetType<std::map<std::string, float> >(),
                                          }),
                DictVectorizerOp);
}  // namespace ML

}  // namespace Lotus
