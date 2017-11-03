#include "core/graph/op.h"

namespace LotusIR {

    REGISTER_OPERATOR_SCHEMA(RNN)
        .Description("Fully-connected RNN where the output is to be fed back to input.")
        .Input(
            "X",
            "Input tensor of shape [S, N, F]",
            "T")
        .Input(
            "W",
            "Weight tensor of shape [M, F]. Used for the linear transformation of the inputs.",
            "T")
        .Input(
            "R",
            "Recurrence weight tensor. Used for the linear transformation of the recurrent state.",
            "T")
        .Input(
            "B",
            "Bias tensor. Used alongside the Weight tensor.",
            "T")
        .Input(
            "Sequence_lens",
            "Tensor vector specifying lengths of the sequences in a batch.",
            "TI")
        .Input(
            "Hidden_init",
            "The initial tensor for hidden state.",
            "T")
        .Output(
            "Y",
            "Output tensor of same type as X, dimensionality defined by the recurrence.",
            "T")
        .TypeConstraint(
            "T",
            { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "TI",
            { "tensor(int32)"},
            "Constrain input and output types to int tensors.")
        .Attr(
            "activation",
            "Activation function such as sigmoid and tanh",
            AttrType::AttributeProto_AttributeType_STRING)
        .Attr(
            "use_bias",
            "Determine whether to use or ignore the bias tensor. Default is false.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "sequence_output",
            "Add additional outputs for the final timestep hidden states, or only "
            "output the final state update. Default is false.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "reverse",
            "everse the inputs. Default is false.",
            AttrType::AttributeProto_AttributeType_INT);

    REGISTER_OPERATOR_SCHEMA(GRU)
        .Description("Gated Recurrent Unit - Cho et al. 2014.")
        .Input(
            "X",
            "Input tensor of shape [S, N, F]",
            "T")
        .Input(
            "W",
            "Weight tensor. Used for the linear transformation of the inputs. "
            "A [3, ] tensor with update, reset and gate weights.",
            "T")
        .Input(
            "B",
            "Bias tensor. Used alongside the Weight tensor. A [3, ] tensor with update, "
            "reset and gate weights.",
            "T")
        .Input(
            "R",
            "Recurrence weight tensor. Used for the linear transformation of the recurrent "
            "state.  A [3, ] tensor with update, reset and gate weights.",
            "T")
        .Input(
            "Sequence_lens",
            "Tensor vector specifying lengths of the sequences in a batch.",
            "TI")
        .Input(
            "Hidden_init",
            "The initial tensor for hidden state.",
            "T")
        .Output(
            "Y",
            "Output tensor of same type as X, dimensionality defined by the recurrence.",
            "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .TypeConstraint("TI", { "tensor(int32)"},
            "Constrain input and output types to int tensors.")
        .Attr(
            "activation",
            "Activation function.",
            AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr(
            "use_bias",
            "Determine whether to use or ignore the bias tensor. Default is false.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "sequence_output",
            "Add additional outputs for the final timestep hidden states, or only "
            "output the final state update. Default is false.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "reverse",
            "everse the inputs. Default is false.",
            AttrType::AttributeProto_AttributeType_INT);


    REGISTER_OPERATOR_SCHEMA(LSTM)
        .Description("Uni-directional long short-term memory (LSTM) layer")
        .Input(
            "X",
            "Input tensor of shape [S, N,F]",
            "T")
        .Input(
            "W",
            "Weight tensor.  Used for the linear transformation of the inputs. A [4, ] "
            "tensor with input, forget, block and output gate weights.",
            "T")
        .Input(
            "B",
            "Bias tensor. Used alongside the Weight tensor. A [4, ] tensor with input, "
            "forget, block and output gate weights.",
            "T")
        .Input(
            "R",
            "Recurrence weight tensor. Used for the linear transformation of the recurrent "
            "state. A [4, ] tensor with input, forget, block and output gate weights.",
            "T")
        .Input(
            "P",
            "Peephole weight tensor. A [3, ] tensor with input, forget and output gate weights.",
            "T")
        .Input(
            "Sequence_lens",
            "Tensor vector specifying lengths of the sequences in a batch.",
            "TI")
        .Input(
            "Hidden_init",
            "The initial tensor for hidden state.",
            "T")
        .Input(
            "Cell_init",
            "The initial tensor for cell state.",
            "T")
        .Output(
            "Y",
            "Output tensor, dimension 0 defined by sequence_output",
            "T")
        .TypeConstraint(
            "T",
            { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "TI",
            { "tensor(int32)"},
            "Constrain input and output types to int tensors.")
        .Attr(
            "activations",
            "A list of activation functions for the gates.",
            AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr(
            "use_bias",
            "Determine whether to use or ignore the bias tensor. Default is true.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "use_peepholes",
            "if True, then use peephole connections in the LSTM",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "sequence_output",
            "If True, output all hidden states. Otherwise only output the final "
            "state update. Default is false.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "reverse",
            "reverse the inputs. Default is false.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "input_forget",
            "Couple the input and forget gates. Default is false",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "clip",
            "Cell clip threshold.",
            AttrType::AttributeProto_AttributeType_FLOAT);


    REGISTER_OPERATOR_SCHEMA(BiLSTM)
        .Description("Bi-directional long short-term memory (LSTM) layer")
        .Input(
            "X",
            "Input tensor of shape [S, N, F]",
            "T")
        .Input(
            "W",
            "Weight tensor.  Used for the linear transformation of the inputs. A [2, 4, ] "
            "tensor with input, forget, block and output gate weights for each of the "
            "forward and backward units.",
            "T")
        .Input(
            "B",
            "Bias tensor.  Used alongside the Weight tensor. A [2, 4, ] tensor with input, "
            "forget, block and output gate weights for each of the forward and backward units.",
            "T")
        .Input(
            "R",
            "Recurrence weight tensor.  Used for the linear transformation of the recurrent "
            "state.  A [2, 4, ] tensor with input, forget, block and output gate weights for "
            "each of the forward and backward units.",
            "T")
        .Input(
            "P",
            "Peephole weight tensor.  A [2, 3, ] tensor with input, forget and output gate "
            "weights for each of the forward and backward units.",
            "T")
        .Input(
            "Sequence_lens",
            "Tensor vector specifying lengths of the sequences in a batch.",
            "TI")
        .Input(
            "Hidden_init",
            "The initial tensor for hidden state.",
            "T")
        .Input(
            "Cell_init",
            "The initial tensor for cell state.",
            "T")
        .Output(
            "Y",
            "Output tensor, dimension 0 defined by sequence_output",
            "T")
        .TypeConstraint(
            "T",
            { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "TI",
            { "tensor(int32)"},
            "Constrain input and output types to int tensors.")
        .Attr(
            "forward_activations",
            "A list of activation functions for forward.",
            AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr(
            "backward_activations",
            "A list of activation functions for backward.",
            AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr(
            "use_bias",
            "Determine whether to use or ignore the bias tensor. Default is true.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "forget_bias",
            "If True, add 1 to the bias of the forget gate at initialization.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "use_peepholes",
            "if True, then use peephole connections in the LSTM",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "sequence_output",
            "If True, output all hidden states. Otherwise only output the final "
            "state update. Default is false.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "input_forget",
            "Couple the input and forget gates. Default is false",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr(
            "clip",
            "Cell clip threshold.",
            AttrType::AttributeProto_AttributeType_FLOAT);
}