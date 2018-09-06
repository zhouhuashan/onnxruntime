#include "expand_dims.h"
#include "onnx/defs/schema.h"
#include "sample.h"

namespace Lotus {
namespace ML {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    ExpandDims,
    1,
    float,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("axis", DataTypeImpl::GetTensorType<int64_t>()),
    ExpandDims);

}  // namespace ML
}  // namespace Lotus
