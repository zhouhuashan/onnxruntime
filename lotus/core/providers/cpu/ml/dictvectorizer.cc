#include "core/providers/cpu/ml/dictvectorizer.h"
namespace Lotus {
namespace ML {
#define REG_MY_KERNEL(T1, T2)                                                          \
  REGISTER_KERNEL(KernelDefBuilder("DictVectorizer")                                   \
                      .Domain(LotusIR::kMLDomain)                                      \
                      .SinceVersion(1)                                                 \
                      .Provider(LotusIR::kCpuExecutionProvider)                        \
                      .TypeConstraint("T1", DataTypeImpl::GetType<std::map<T1, T2>>()) \
                      .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>()),        \
                  DictVectorizerOp<T1, T2>);

#define REG_MY_STRING_KERNEL(T) REG_MY_KERNEL(std::string, T);
#define REG_MY_INT64_KERNEL(T) REG_MY_KERNEL(int64_t, T);

REG_MY_STRING_KERNEL(int64_t);
REG_MY_STRING_KERNEL(float);
REG_MY_STRING_KERNEL(double);

REG_MY_INT64_KERNEL(std::string);
REG_MY_INT64_KERNEL(float);
REG_MY_INT64_KERNEL(double);
}  // namespace ML

}  // namespace Lotus
