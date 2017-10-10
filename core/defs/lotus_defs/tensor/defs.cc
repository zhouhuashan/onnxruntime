#include "core/graph/op.h"

namespace LotusIR {
    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Flatten)
        .Description("Flattens the input tensor into a 2D matrix, "
            "keeping the first dimension unchanged.")
        .Input("input", "A tensor of rank >= 2.", "T")
        .Output("output", "A tensor of rank 2 with the contents of the input tensor, "
            "with first dimension equal first dimension of input, and remaining "
            "input dimensions flatenned into the inner dimension of the output.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Reshape)
        .Description("Reshape the input tensor similar to numpy.reshape. "
            "                                                                                    "
            "It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor. "
            "                                                                             "
            "At most one dimension of the new shape can be -1. In this case, the value is "
            "inferred from the size of the tensor and the remaining dimensions. A dimensions "
            "could also be 0, in which case the actual dimension value is going to be copied "
            "from the shape argument.")
        .Input("data", "An input tensor.", "T")
        .Output("reshaped", "Reshaped data.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("shape", "Tensor of shape declarations for the output. Must be compatible with "
            "the input. At most one dimension of the new shape can be -1. In this case, the "
            "value is inferred from the size of the tensor and the remaining dimensions. A "
            "dimension could also be 0, in which case the actual dimension value is going to "
            "be copied from the input tensor.", AttrType::INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Split)
        .Description("Split a tensor into a list of tensors, along the specified 'axis'. "
            "The lengths of the split can be specified using argument 'axis' or "
            "optional second input blob to the operator. Otherwise, the tensor is split "
            "to equal sized parts.")
        .Input("input", "The tensor to split", "T")
        .Input("split", "Optional list of output lengths (see also arg 'split')", "T")
        .Output("output", "A list of output tensors", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("axis", "Which axis to split on", AttrType::INT)
        .Attr("split", "Number of tensors to output.", AttrType::INT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Transpose)
        .Description("Transpose the input tensor similar to numpy.transpose. For example, "
            "when axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape "
            "will be (2, 1, 3).")
        .Input("data", "An input tensor.", "T")
        .Output("transposed", "Transposed output.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("perm", "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given.", AttrType::INTS);

    REGISTER_OPERATOR_SCHEMA(RepeatElements)
        .Description("Repeat the elements of a tensor along an axis.")
        .Input("input", "An input tensor.", "T")
        .Output("output", "Repeated output.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("perm", "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given.", AttrType::INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Concat)
        .Description("Concatenate takes as input a list of tensors, all of the same shape"
            "expect for the concatenation axis, and returns a single tensor, the concatenation"
            "of all inputs.")
        .Input("input", "A list of input tensors.", "T")
        .Output("output", "Concatenated tensor", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("axis", "Axis along which to concatenate", AttrType::INT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Slice)
        .Description("Produces a slice of the input tensor along multiple axes. Similar to "
            "numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html "
            "                                                                              "
            "Slices are passed as two keyword argument lists with starting and end indices "
            "for each dimension of the input `data` tensor. If a negative value is passed "
            "for any of the start or end indices, it represent number of elements before "
            "the end of that dimension. "
            "                                                                            "
            "`strides` is the  step sizes when applying slicing, negative value means in "
            "reverse order.")
        .Input("input", "Tensor of data to extract slices from.", "T")
        .Output("output", "Sliced data tensor.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("starts", "List of starting indices", AttrType::INTS)
        .Attr("ends", "List of ending indices", AttrType::INTS);

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(BatchToSpace)
        .Description("BatchToSpace for 4-D tensors of type T. "
            "Rearranges (permutes) data from batch into blocks of spatial data, "
            "followed by cropping. This is the reverse transformation of "
            "SpaceToBatch. More specifically, this op outputs a copy of the input "
            "tensor where values from the batch dimension are moved in spatial "
            "blocks to the height and width dimensions, followed by cropping along "
            "the height and width dimensions.")
        .Input("input", "Input tensor of [N,C,H,W]", "T")
        .Output("output", "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, "
            "W * blocksize]", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("blocksize", "Blocks of [blocksize,blocksize] are moved.", AttrType::INT);

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(SpaceToBatch)
        .Description("SpaceToBatch for 4-D tensors of type T. "
            "Zero-pads and then rearranges (permutes) blocks of spatial data into "
            "batch. More specifically, this op outputs a copy of the input tensor "
            "where values from the height and width dimensions are moved to the "
            "batch dimension. After the zero-padding, both height and width of the "
            "input must be divisible by the block size.")
        .Input("input", "Input tensor of [N,C,H,W]", "T")
        .Output("output", "Output tensor of [N, C * blocksize * blocksize, H/blocksize, "
            "W/blocksize]", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("blocksize", "Blocks of [blocksize,blocksize] are moved.", AttrType::INT);

}