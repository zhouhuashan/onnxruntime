#include "core/graph/op.h"
#include "onnx/defs/schema.h"

namespace LotusIR {

ONNX_OPERATOR_SCHEMA(LocalResponseNormalization)
    .Attr("size", "The number of channels to sum over", AttributeProto::INT)
    .Attr("alpha", "Scaling parameter", AttributeProto::FLOAT)
    .Attr("beta", "The exponent", AttributeProto::FLOAT)
    .Attr("bias", "Default to 1.f", AttributeProto::FLOAT, 1.0f)
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output "
        " types to float tensors.")
    .SetDoc(R"DOC(
Local Response Normalization. It normalizes over local input regions.
Each input value is divided by
(bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta.
)DOC");

// Taken from RS4
ONNX_OPERATOR_SCHEMA(MeanSubtraction)
    .SetDoc("Subtracts the provided mean image from the input image.")
    .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
    .Output(1, "output", "Result, has same shape and type as X", "T")
    .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                    "Constrain input and output types to float tensors.")
    .Attr("image", "Image tensor stored as a sequence of floats [C,H,W].", AttrType::AttributeProto_AttributeType_TENSOR);

// Take from RS4
ONNX_OPERATOR_SCHEMA(Embedding)
.SetDoc("Turns positive integers (indexes) into dense vectors of fixed size. "
    "eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] "
    "TODO: Omits use of CoreML bias parameter.")
    .Input(0, "input", "1-D tensor of integers representing indices in the embedding "
        "dictionary with length [N] and values [0, input_dim -1]", "T1")
    .Input(1, "W", "2-D tensor of weights [O, I]", "T2")
    .Output(0, "output", "Output tensor of computed features [N, O].", "T2")
    .TypeConstraint("T1", { "tensor(uint64)" }, "Constrain input types to ints.")
    .TypeConstraint("T2", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain output types to float tensors.")
    .Attr("input_dim", "Size of the input vocabulary.", AttrType::AttributeProto_AttributeType_INT)
    .Attr("output_dim", "Dimension of the embedding output vectors.", AttrType::AttributeProto_AttributeType_INT);
}  // namespace LotusIR
