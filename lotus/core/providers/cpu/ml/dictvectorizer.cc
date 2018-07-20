#include "core/providers/cpu/ml/dictvectorizer.h"

using namespace std;

namespace Lotus {
namespace ML {

#define REG_NAMED_KERNEL(name, T1, T2)                                                 \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                   \
    DictVectorizer,                                                                    \
    1,                                                                                 \
    name,                                                                       \
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetType<std::map<T1, T2>>()) \
                      .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>()),        \
    DictVectorizerOp<T1, T2>);

#define REG_MY_KERNEL(T1, T2) REG_NAMED_KERNEL(T1##_##T2, T1, T2)

#define REG_MY_STRING_KERNEL(T) REG_MY_KERNEL(string, T);
#define REG_MY_INT64_KERNEL(T) REG_MY_KERNEL(int64_t, T);

REG_MY_STRING_KERNEL(int64_t);
REG_MY_STRING_KERNEL(float);
REG_MY_STRING_KERNEL(double);

REG_MY_INT64_KERNEL(string);
REG_MY_INT64_KERNEL(float);
REG_MY_INT64_KERNEL(double);

}  // namespace ML

}  // namespace Lotus
