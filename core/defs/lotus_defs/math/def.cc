#include "core/graph/op.h"

namespace LotusIR {

    #define REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(OpName)                                                    \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                        \
        .Description("Elementwise "#OpName" takes one or more input data (Tensor<T>) and produces one "     \
            "output data (Tensor<T>) where the declared function is applied to the input "                  \
            "tensors elementwise.")                                                                         \
        .Input("data_0", "First of the input tensors. Can be inplace.", "T")                                \
        .Output("output", "Output tensor. Same dimension as inputs.", "T")                                  \
        .TypeConstraint("T", { "float16", "float", "double" },                                              \
            "Constrain input and output types to floats.");

    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Add)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Sub)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Mul)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Div)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Max)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Min)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Sum)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Mean)

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Neg)
        .Description("Neg takes one input data (Tensor<T>) and produces one output data \
            (Tensor<T>) where each element flipped sign, y = -x, is applied to \
            the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Abs)
        .Description("Absolute takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the absolute is, y = abs(x), is applied to "
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
             "Constrain input and output types to floats.");

    // Take from ONNX
    REGISTER_OPERATOR_SCHEMA(Reciprocal)
        .Description("Reciprocal takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the reciprocal is, y = 1/x, is applied to "
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Floor)
        .SetDoc("Floor takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the floor is, y = floor(x), is applied to "
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Ceil)
        .Description("Ceil takes one input data (Tensor<T>) and produces one output data"
            "(Tensor<T>) where the ceil is, y = ceil(x), is applied to"
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(Clip)
        .Description("Clip operator limits the given input within an interval. "
            "The interval is specified with arguments 'min' and 'max'. They default to "
            "numeric_limits::lowest() and numeric_limits::max() respectively. The clipping "
            "operation can be done in in-place fashion too, where the input and output blobs "
            "are the same.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("min", "Minimum value, under which element is replaced by min", AttrType::FLOAT)
        .Attr("max", "Maximum value, under which element is replaced by max", AttrType::FLOAT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Sqrt)
        .Description("Square root takes one input data (Tensor<T>) and produces one output "
            "data Tensor<T>) where the square root is, y = x^0.5, is applied to "
            "the tensor elementwise. If x is negative, then it will return NaN.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Exp)
        .Description("Calculates the exponential of the given input tensor, element-wise. "
            "This operation can be done in an in-place fashion too, by providing the same "
            "input and output blobs.")
        .Input("input", "input tensor", "T")
        .Output("output", "The exponential of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Log)
        .Description("Calculates the natural log of the given input tensor, element-wise. "
            "This operation can be done in an in-place fashion too, by providing the same "
            "input and output blobs.")
        .Input("input", "input tensor", "T")
        .Output("output", "The natural  log of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Pow)
        .Description("Pow takes input data (Tensor<T>) and an argument exponent, and "
            "produces one output data (Tensor<T>) where the function `f(x) = x^exponent`, "
            "is applied to the data tensor elementwise.")
        .Input("input", "input tensor", "T")
        .Output("output", "The x^exponent value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("exponent", "The exponent of the power function.", AttrType::FLOAT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Dot)
        .Description("Apply dot product between 2 tensors. Similar to numpy implementation: "
            "https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html")
        .Input("X", "Input tensor of any shape", "T")
        .Input("Y", "Input tensor of any shape", "T")
        .Output("output", "Output tensor the dot product between X and Y.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");
}