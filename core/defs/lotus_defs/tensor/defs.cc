#include "op.h"

namespace LotusIR {
    
    REGISTER_OPERATOR_SCHEMA(Scale)
        .Description("Scale takes one input data (Tensor<float>) and produces one output data"
            "(Tensor<float>) whose value is the input data tensor scaled element-wise.")
        .Input("X", "input tensor", "T")
        .Output("C", "Result, has same shape and type as X", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output types to floats.")
        .Attr("scale", "the scale to apply, default 1.0", AttrType::FLOAT, float(1.0))
        .Attr("bias", "the scale to apply, default 1.0", AttrType::FLOAT, float(1.0));

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
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and "
            "output types to floats.")
        .Attr("blocksize", "Blocks of [blocksize,blocksize] are moved.", AttrType::INT);

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
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and "
            "output types to floats.")
        .Attr("blocksize", "Blocks of [blocksize,blocksize] are moved.", AttrType::INT);

}