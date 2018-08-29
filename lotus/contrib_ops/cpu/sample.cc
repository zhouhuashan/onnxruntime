#include "sample.h"
#include "onnx/defs/schema.h"

using namespace onnx;

namespace Lotus {
namespace ML {
// These ops are internal-only, so register outside of onnx
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    SampleOp,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    SampleOp<float>);
}  // namespace ML
}  // namespace Lotus
