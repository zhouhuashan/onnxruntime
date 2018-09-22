// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/contrib_ops.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"

#include "./cpu/attnlstm/attn_lstm_schema_defs.h"

namespace onnxruntime {
namespace ml {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OPTIONAL;
using ::ONNX_NAMESPACE::OpSchema;

void RegisterContribSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(SampleOp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          ONNX_NAMESPACE::OpSchema::numeric_types_for_math_reduction(),
          "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetDoc(R"DOC(
Sample echo operator.)DOC");

  // register schemas for more operators here

  ONNX_CONTRIB_OPERATOR_SCHEMA(ExpandDims)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Input(1, "axis", "Specified axis to insert a dimension", "tensor(int32)")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          ONNX_NAMESPACE::OpSchema::all_tensor_types(),
          "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetDoc(R"DOC(ExpandDims echo operator.)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(AttnLSTM, RegisterAttnLSTMContribOpSchema);
}

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, SampleOp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, ExpandDims);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, AttnLSTM);

void RegisterContribKernels(std::function<void(KernelCreateInfo&&)> fn) {
  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, SampleOp)>());

  // add more kernels here

  fn(BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, ExpandDims)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, AttnLSTM)>());
}
}  // namespace ml
}  // namespace onnxruntime
