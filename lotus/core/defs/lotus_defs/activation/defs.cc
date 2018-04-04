#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "core/graph/constants.h"

namespace LotusIR {
Status MsOpRegistry::RegisterMsActivationOps() {
  ONNX_OPERATOR_SCHEMA(ParametericSoftplus)
      .SetDomain(kMSDomain)
      .SetDoc(
          "Softplus takes input data (Tensor<T>) and parametric tensors, "
          "producing one output data (Tensor<T>) where the function, "
          "y = alpha * log(1 + exp(beta * x), is applied to the tensor elementwise.")
      .Input(0, "X", "Input tensor, typically 1-D.", "T")
      .Output(0, "Y", "Output tensor of same shape and type as input X.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                      "Constrain input and output types to float tensors.")
      .Attr("alpha",
            "Alpha tensor. If `alpha` is of size 1, "
            "the value is shared across different channels.",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.0))
      .Attr("beta",
            "Beta tensor. If `beta` is of size 1, "
            "the value is shared across different channels.",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.0));
  return Status::OK();
}
// Taken from RS4

}  // namespace LotusIR
