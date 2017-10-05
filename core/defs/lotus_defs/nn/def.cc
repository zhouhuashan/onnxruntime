#include "op.h"

namespace LotusIR {

    REGISTER_OPERATOR_SCHEMA(FC)
        .Description("Computes the result of passing an input vector X into a fully"
        "connected layer with 2D weight matrix W and 1D bias vector b.That is, "
        "the layer computes Y = X * W^T + b, where X has size(M x K), "
        "W has size(N x K), b has size(N), and Y has size(M x N), "
        "where M is often the batch size.")
        .Input("X", "input tensor that's coerced into a 2D matrix of size (MxK) ", "T")
        .Input("W", "A tensor that is coerced into a 2D blob of size (KxN) containing fully connected weight matrix", "T")
        .Input("b", "1D blob containing bias vector", "T")
        .Output("Y", "output tensor", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output types to floats.")
        .Attr("map_rank", "Optional, number of leading dimensions that are not transformed by the operator",
            AttrType::INT)
        .Attr("activation", "Activation function to be computed with operator. ", AttrType::FLOAT);

    REGISTER_OPERATOR_SCHEMA(ROIPooling)
        .Description("Carries out ROI Pooling for Faster-RCNN.")
        .Input("X", "The input 4-D tensor of data. Only NCHW order is currently supported.", "T")
        .Input("rois", "RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of "
        "shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].", "T")
        .Output("Y", "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_h, pooled_w).", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output types to floats.")
        .Attr("spatial_scale", "Multiplicative spatial scale factor to translate ROI coordinates from their "
        "input scale to the scale used when pooling (Default: 1.0).", AttrType::FLOAT, float(1.0))
        .Attr("roi_output_shape", "Dimensions (width x height) of the ROI pooling output shape.", AttrType::INTS );

    REGISTER_OPERATOR_SCHEMA(L2Pooling)
        .Description("Carries out L2 Pooling.")
        .Input("X", "Input tensor of any shape", "T")
        .Output("Y", "Output tensor of same shape and type as input X", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output types to floats.")
        .Attr("kernel_size", "receptive field (window) to pool over, e.g. (2,2) (not including the input feature-map dept) ", AttrType::INTS)
        .Attr("strides", "increment when sliding the pool over the input. E.g. (2,2) to reduce the dimensions by 2", AttrType::INTS)
        .Attr("pad", "if False (default), then the pool will be shifted over the “valid” area of input, that is,"
            "no value outside the area is used. If pad is True on the other hand, the pool will be applied to all input positions, "
            "and values outside the valid region will be considered zero. For average pooling,"
            "count for average does not include padded values.", AttrType::INT, int64_t(0))
        .Attr("global", "If true, use global pooling.", AttrType::INT );

    REGISTER_OPERATOR_SCHEMA(LocalResponseNormalization)
        .Description("Perform local response normalization. "
            "NOTE: Only supports Caffe across channel mode. ")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
             " types to floats.")
        .Attr("local_radius", "[default 5]: the number of channels to sum over (for cross "
              "channel LRN) or the side length of the square region to sum over (for within "
              "channel LRN)", AttrType::INT, int64_t(5))
        .Attr("alpha", "Scalar scaling factor. Default is 0.0001", AttrType::FLOAT, float(0.0001))
        .Attr("beta", "Scalar exponent in the LRN.  Default is 0.5.", AttrType::FLOAT, float(0.5))
        .Attr("beta", "An offset (must be positive to avoid dividing by 0). Defaults to 1.0.",
            AttrType::FLOAT, float(1.0));

    REGISTER_OPERATOR_SCHEMA(MeanVarianceNormalization)
        .Description("Perform mean variance normalization.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output "
            "types to floats.")
        .Attr("across_channels", "If true, mean and variance are computed across channels. "
            "Default is false.", AttrType::INT, int64_t(0))
        .Attr("normalize_variance", "If false, normalize the mean only. Default is true.",
            AttrType::INT, int64_t(1)); 

    REGISTER_OPERATOR_SCHEMA(L2Normalization)
        .Description("Perform L2 normalization  Divide each element by the square root of the "
            "sum of squares of all elements in the input tensor.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "float16", "float", "float" }, "Constrain input and output "
            "types to floats.")
        .Attr("Axis", "Axis along which to perform normalization.", AttrType::INT);

    REGISTER_OPERATOR_SCHEMA(Embedding)
        .Description("Turns positive integers (indexes) into dense vectors of fixed size. "
            "eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] "
            "TODO: Omits use of CoreML bias parameter.")
        .Input("input", "1-D tensor of integers representing indices in the embedding "
            "dictionary with length [N] and values [0, input_dim -1]", "T1")
        .Output("output", "Output tensor of computed features [N, O].", "T2")
        .TypeConstraint("T1", { "uint64" }, "Constrain input types to ints.")
        .TypeConstraint("T2", { "float16", "float", "double" },
                "Constrain output types to floats.")
        .Attr("input_dim", "Size of the input vocabulary.", AttrType::INT)
        .Attr("output_dim", "Dimension of the embedding output vectors.", AttrType::INT)
        .Attr("weights", "2-D tensor of weights [O,I]", AttrType::FLOAT);

    REGISTER_OPERATOR_SCHEMA(Upsample)
        .Description("Scale up spatial dimensions.  Use interpolation to fill in values")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same shape and type as X", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and "
            "output types to floats.")
        .Attr("mode", "enum {'NN', 'BILINEAR' }, Nearest neighbor or bilinear upsampling.",
            AttrType::STRING)
        .Attr("sacle", "1-D scale factor tensor with values for (H,W)", AttrType::INT);

    REGISTER_OPERATOR_SCHEMA(RepeatElements)
        .Description("Scale up spatial dimensions.  Use interpolation to fill in values")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Result, has same shape and type as X", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and "
            "output types to floats.")
        .Attr("axis", "Axis along which to repeat. Default is 0.",AttrType::INT, int64_t(0))
        .Attr("repetitions", "Number of repeated copies to make of the input tensor.",
            AttrType::INT);

    REGISTER_OPERATOR_SCHEMA(Crop)
        .Description("Crop and image to the specified spatial dimensions.  If scale is given,"
            "then optionally start the crop offset by the left/top border amounts.  "
            "If scale is not provided, crop the borders as provided.")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same type as X, with H and W dimensions reduced.", "T")
        .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and "
            "output types to floats.")
        .Attr("border", "A 1-D tensor of values (leftBorder, topBorder, rightBorder, bottomBorder)",
            AttrType::INT)
        .Attr("scale", "A 1-D tensor of values (height, width)", AttrType::INT);
}

