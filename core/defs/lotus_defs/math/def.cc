#include "op.h"

namespace LotusIR {

    REGISTER_OPERATOR_SCHEMA(Linear)
        .Description("Linear takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the linear function, f(x)= alpha * x + beta is "
            "applied to the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("alpha", "Scalar multiplication factor", AttrType::FLOAT)
        .Attr("beta", "Scalar offset", AttrType::FLOAT);

    REGISTER_OPERATOR_SCHEMA(HardSigmoid)
        .Description("HardSigmoid takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the hard sigmoid function, f(x) = max⁡(0,min⁡(alpha*x+beta,1)), "
            "is applied to the  tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("alpha", "Scaling value", AttrType::FLOAT)
        .Attr("beta", "Scalar offset", AttrType::FLOAT);

    REGISTER_OPERATOR_SCHEMA(ScaledTanh)
        .Description("ScaledTanh takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the scaled hyperbolic tangent function, "
            "f(x) = alpha*tanh⁡(beta*x), is applied to the  tensor elementwise.")
        .Input("input", "Input tensor, typically 1-D.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("alpha", "Scaling value", AttrType::FLOAT)
        .Attr("beta", "Scaling value", AttrType::FLOAT);

    REGISTER_OPERATOR_SCHEMA(ThresholdedReLU)
        .Description("Thresholded Relu takes input data (Tensor<T>) and threshold as input, and "
            "produces one output data (Tensor<T>) where the function `f(x) = 0 for x < alpha, "
            "x for x >= alpha`, is applied to the data tensor elementwise.")
        .Input("input", "Input tensor, typically 1-D.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("alpha", "Scalar threshold value", AttrType::FLOAT);

    REGISTER_OPERATOR_SCHEMA(LogSoftmax)
        .Description("Log Softmax takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the function, y = log(1 / sum(exp(X)) * exp(x)), is applied "
            "to the tensor elementwise.")
        .Input("input", "Input tensor of shape [N, F].", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.");

    REGISTER_OPERATOR_SCHEMA(Hardmax)
        .Description("Compute the hardmax normalized values for each layer in the batch "
            "of the given input. The input is a 2-D tensor (Tensor<float>) of size "
            "(batch_size x input_feature_dimensions). The output tensor has the same shape "
            "and contains the softmax normalized values of the corresponding input. "
            "\n"                                                                                                                  
            "X does not need to explicitly be a 2D vector; rather, it will be coerced into "
            "one. For an arbitrary n-dimensional tensor X in [a_0, a_1, ..., a_{k-1}, "
            "a_k, ..., a_{n-1}] and k is the axis provided, then X will be coerced into a "
            "2-dimensional tensor with dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. "
            "For the default case where axis=1, this means the X tensor will be coerced into "
            "a 2D tensor of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the "
            "batch size.  In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D. "
            "Each of these dimensions must be matched correctly, or else the operator will "
            "throw errors.")
        .Input("input", "Input tensor of shape [N, F].", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("axis", "Default to 1; describes the axis of the inputs when coerced to 2D; "
            "defaults to one because the 0th axis most likely describes the batch size.",
            AttrType::INT, int64_t(1));

    REGISTER_OPERATOR_SCHEMA(Softsign)
        .Description("Softsign takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the function, y = x / (1 + abs(x)), is applied to the "
            "tensor elementwise.")
        .Input("input", "Input tensor, typically 1-D.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("alpha", "Coefficient of SELU default to 1.6732.", AttrType::FLOAT, float(1.6732));

    REGISTER_OPERATOR_SCHEMA(Softplus)
        .Description("Softplus takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the function, y = log(1 + exp(beta * x)) / beta, is "
            "applied to the tensor elementwise.  When steepness is greater than 1, the "
            "function is y = log(1 + exp(beta * steepness * x)) / steepness.")
        .Input("input", "Input tensor, typically 1-D.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("beta", "Scaling value, default to 1.0.", AttrType::FLOAT, float(1.0))
        .Attr("steepness", "Steepness factor, default is 1.0", AttrType::FLOAT, float(1.0));

    REGISTER_OPERATOR_SCHEMA(ParametericSoftplus)
        .Description("Softplus takes input data (Tensor<T>) and parametric tensors, "
            "producing one output data (Tensor<T>) where the function, "
            "y = alpha * log(1 + exp(beta * x), is applied to the tensor elementwise.")
        .Input("input", "Input tensor, typically 1-D.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("alpha", "Alpha tensor. If `alpha` is of size 1, "
            "the value is shared across different channels.", AttrType::FLOAT, float(1.0))
        .Attr("beta", "Beta tensor. If `beta` is of size 1, "
            "the value is shared across different channels.", AttrType::FLOAT, float(1.0));

    REGISTER_OPERATOR_SCHEMA(Identity)
        .Description("Identity takes one input data (Tensor<T>) and produces one "
            "output data (Tensor<T>) where the function, y = x, is applied to the "
            "tensor elementwise.")
        .Input("input", "input tensor", "T")
        .Output("output", "output tensor", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.");

}