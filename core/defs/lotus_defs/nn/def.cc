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
        .Attr("map_rank", "Optional, number of leading dimensions that are not transformed by the operator", AttrType::INT, false)
        .Attr("activation", "Activation function to be computed with operator. ", AttrType::FLOAT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Conv)
        .Description("The convolution operator consumes an input tensor and a filter, and"
            "computes the output.")
        .Input("X",
             "Input data tensor from previous layer; has size (N x C x H x W)"
             ", where N is the batch size, C is the number of channels, and"
             " H and W are the height and width. Note that this is for the 2D image."
             "Otherwise the size is (N x D1 x D2 ... x Dn)",
             "T")
        .Input("weights",
             "The weight tensor that will be used in the convolutions; has size (M x C x kH x kW), "
             "where C is the number of channels, and kH and kW are the height and width of the kernel, "
             "and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be "
             "(M x C x k1 x k2 x ... x kn), where is the dimension of the kernel",
             "T")
        .Output("Y",
              "Output data tensor that contains the result of the convolution. The "
              "output dimensions are functions of the kernel size, stride size, "
              "and pad lengths.",
              "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("kernel_shape",
            "The shape of the convolution kernel.",
             AttrType::INTS)
        .Attr("dilations",
            "dilation value along each axis of the filter.",
            AttrType::INTS)
        .Attr("strides",
            "stride along each axis.",
            AttrType::INTS)
        .Attr("pads",
            "Padding along each axis, can take the value 0 (False) or non 0 (True)",
            AttrType::INTS)
        .Attr("group",
            "number of groups input channels and output channels are divided into",
            AttrType::INT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(ConvTranspose)
        .Description("The convolution transpose operator consumes an input tensor and a filter,"
            "and computes the output.")
        .Input("X",
             "Input data tensor from previous layer; has size (N x C x H x W)"
             ", where N is the batch size, C is the number of channels, and"
             " H and W are the height and width. Note that this is for the 2D image."
             "Otherwise the size is (N x D1 x D2 ... x Dn)",
             "T")
        .Input("weights",
             "The weight tensor that will be used in the convolutions; has size (M x C x kH x kW), "
             "where C is the number of channels, and kH and kW are the height and width of the kernel, "
             "and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be "
             "(M x C x k1 x k2 x ... x kn), where is the dimension of the kernel",
             "T")
        .Output("Y",
              "Output data tensor that contains the result of the convolution. The "
              "output dimensions are functions of the kernel size, stride size, "
              "and pad lengths.",
              "T")
		.TypeConstraint("T", { "float16", "float", "double" },
			"Constrain input and output types to floats.")
        .Attr("kernel_shape",
            "The shape of the convolution kernel.",
             AttrType::INTS)
        .Attr("output_shape",
            "The shape of the output.",
            AttrType::INTS)
        .Attr("dilations",
            "dilation value along each axis of the filter.",
            AttrType::INTS)
        .Attr("strides",
            "stride along each axis.",
            AttrType::INTS)
        .Attr("pads",
            "Padding along each axis, can take the value 0 (False) or non 0 (True)",
            AttrType::INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(AveragePool)
        .Description("AveragePool consumes an input tensor X and applies average pooling across the"
            "the tensor according to kernel sizes, stride sizes, and pad lengths."
            "Average pooling consisting of averaging all values of a subset of the"
            "input tensor according to the kernel size and downsampling the"
            "data into the output tensor Y for further processing.")
        .Input("X",
            "Input data tensor from the previous operator; dimensions for image case "
            "are (N x C x H x W), where N is the batch size, C is the number of channels, "
            "and H and W are the height and the width of the data. For non image case, the "
            "dimension are in the form of (N x D1 x D2 ... Dn), where N is the batch size.",
            "T")
        .Output("Y",
            "Output data tensor from average pooling across the input tensor. "
            "Dimensions will vary based on various kernel, stride, and pad sizes.")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("kernel_shape",
            "The size of the kernel along each axis.",
            AttrType::INTS)
        .Attr("pads",
            "Padding along each axis, can take the value 0 (False) or non 0 (True)",
            AttrType::INTS)
        .Attr("strides",
            "Stride along each axis.",
            AttrType::INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(MaxPool)
        .Description("MaxPool consumes an input tensor X and applies max pooling across the"
            "the tensor according to kernel sizes, stride sizes, and pad lengths."
            "Average pooling consisting of averaging all values of a subset of the"
            "input tensor according to the kernel size and downsampling the"
            "data into the output tensor Y for further processing.")
        .Input("X",
            "Input data tensor from the previous operator; dimensions for image case "
            "are (N x C x H x W), where N is the batch size, C is the number of channels, "
            "and H and W are the height and the width of the data. For non image case, the "
            "dimension are in the form of (N x D1 x D2 ... Dn), where N is the batch size.",
            "T")
        .Output("Y",
            "Output data tensor from max pooling across the input tensor. "
            "Dimensions will vary based on various kernel, stride, and pad sizes.",
            "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("kernel_shape",
            "The size of the kernel along each axis.",
            AttrType::INTS)
        .Attr("strides",
            "Stride along each axis.",
            AttrType::INTS)
        .Attr("pads",
            "Padding along each axis, can take the value 0 (False) or non 0 (True)",
            AttrType::INTS)
        .Attr("dilations",
            "Dilaton along each axis, 1 mean no dilation.",
            AttrType::INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(BatchNormalization)
        .Description("Carries out batch normalization as described in the paper"
            "https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,"
            "there are multiple cases for the number of outputs, which we list below:"
            ""
            "Output case #1: Y, mean, var, saved_mean, saved_var (training mode)"
            "Output case #2: Y (test mode)")
        .Input("X",
            "The input 4-dimensional tensor of shape NCHW or NHWC depending "
            "on the order parameter.",
            "T")
        .Input("scale",
            "The scale as a 1-dimensional tensor of size C to be applied to the "
            "output.",
            "T")
        .Input("bias",
            "The bias as a 1-dimensional tensor of size C to be applied to the "
            "output.",
            "T")
        .Input("mean",
            "The running mean (training) or the estimated mean (testing) "
            "as a 1-dimensional tensor of size C.",
            "T")
        .Input("var",
            "The running variance (training) or the estimated "
            "variance (testing) as a 1-dimensional tensor of size C.",
            "T")
        .Output("Y", "The output 4-dimensional tensor of the same shape as X.",
            "T")
        .Output("mean",
            "The running mean after the BatchNormalization operator. Must be in-place "
            "with the input mean. Should not be used for testing.",
            "T")
        .Output("var",
            "The running variance after the BatchNormalization operator. Must be "
            "in-place with the input var. Should not be used for testing.",
            "T")
        .Output("saved_mean",
            "Saved mean used during training to speed up gradient "
            "computation. Should not be used for testing.",
            "T")
        .Output("saved_var",
            "Saved variance used during training to speed up "
            "gradient computation. Should not be used for testing.",
            "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttrType::FLOAT)
        .Attr("is_test",
            "If set to nonzero, run spatial batch normalization in test mode.",
            AttrType::INT)
        .Attr("momentum",
            "Factor used in computing the running mean and variance."
            "e.g., running_mean = running_mean * momentum + mean * (1 - momentum)",
            AttrType::FLOAT)
        .Attr("spatial",
            "Compute the mean and variance across all spatial elements or per feature.",
            AttrType::INT);

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

    // Take from RS4
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
        .Attr("weights", "2-D tensor of weights [O,I]", AttrType::FLOATS);

    // Taken from RS4
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
        .Attr("axis", "Axis along which to repeat. Default is 0.", AttrType::INT, int64_t(0))
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

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(Padding)
        .Description("Perform padding along spatial dimensions of an image.")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same type as X, with H and W extended by the  \
            padding amounts.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("border", "A 1-D tensor of values (leftBorder, topBorder, rightBorder, bottomBorder)",
            AttrType::INT)
        .Attr("constant", "Constant padding value.", AttrType::FLOAT)
        .Attr("mode", "Method to use when padding: ‘CONSTANT’, ‘REFLECT’, ‘REPLICATE’;"
            "Constant padding simply fills in the values created by the border. "
            "Reflect takes a reflection of the values at the border into the padding space."
            "Replicate copies the border, projecting into the padding space.",
            AttrType::STRING);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(MeanSubtraction)
        .Description("Subtracts the provided mean image from the input image.")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same shape and type as X", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("image", "Image tensor stored as a sequence of floats [C,H,W].", AttrType::TENSOR);

    REGISTER_OPERATOR_SCHEMA(Constant)
        .Description("A constant tensor.")
        .Output("output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", { "float16", "float", "double" },
            "Constrain input and output types to floats.")
        .Attr("value", "The value for the elements of the output tensor.", AttrType::TENSOR);
}

