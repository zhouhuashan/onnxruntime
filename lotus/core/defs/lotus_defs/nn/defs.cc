#include "core/graph/op.h"
#include "onnx/defs/schema.h"

namespace LotusIR {
Status MsOpRegistry::RegisterMsNNOps() {
  // Taken from RS4
  ONNX_OPERATOR_SCHEMA(MeanSubtraction)
      .SetDomain(kMSDomain)
      .SetDoc("Subtracts the provided mean image from the input image.")
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as X", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                      "Constrain input and output types to float tensors.")
      .Attr("image", "Image tensor stored as a sequence of floats [C,H,W].", AttrType::AttributeProto_AttributeType_TENSOR);

  return Status::OK();
}
}  // namespace LotusIR
