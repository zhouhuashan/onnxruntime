#include "core/providers/cpu/controlflow/scan.h"

#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnx;
using namespace Lotus::Common;

namespace Lotus {

onnx::OpSchema Scan::GetScanOpSchema() {
  // Preliminary definition
  onnx::OpSchema schema("Scan", __FILE__, __LINE__);
  schema.SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("<insert scan loop doc>")
      .Input(0,
             "initial_state_and_scan_inputs",
             "Initial values of the loop's N state variables followed by M scan_inputs",
             "V",
             OpSchema::Variadic)
      .Output(0,
              "final_state_and_scan_outputs",
              "Final values of the loop's N state variables followed by K scan_outputs",
              "V",
              OpSchema::Variadic)
      .Attr("body",
            "The graph run each iteration. It has N+M inputs: "
            "(loop state variables..., scan_input_elts...). It has N+K outputs: "
            "(loop state variables..., scan_output_elts...). Each "
            "scan_output is created by concatenating the value of the specified "
            "scan_output_elt value at the end of each iteration of the loop. It is an error"
            " if the dimensions of these values change across loop iterations.",
            AttributeProto::GRAPH,
            true)
      .Attr("scan_axes",
            "A list of M axes. The i-th element of the list specifies the axis "
            "to be scanned for the i-th scan_input tensor.",
            AttributeProto::INTS)
      .SinceVersion(7)
      .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types");

  return schema;
}

Status Scan::Compute(OpKernelContext* ctx) const {
  auto ctx_impl = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_impl->SubgraphSessionState("body");
  LOTUS_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");

  // these are the num inputs and outputs according to the schema
  auto num_inputs = ctx->InputCount();
  auto num_outputs = ctx->OutputCount();

  LOTUS_ENFORCE(num_inputs > 0 && num_outputs >= 0);

  auto& graph = *session_state->GetGraph();
  auto& graph_inputs = graph.GetInputs();
  auto& graph_outputs = graph.GetOutputs();

  auto num_variadic_inputs = ctx->NumVariadicInputs(0);
  auto num_variadic_outputs = ctx->OutputCount();

  LOTUS_ENFORCE(num_variadic_inputs == graph_inputs.size());
  LOTUS_ENFORCE(num_variadic_outputs == gsl::narrow_cast<int>(graph_outputs.size()));

  // Rough initial pieces that will handle calling the subgraph once.
  // TODO: Determine which are loop variables vs inputs/outputs
  //       Setup shape/type inferencing for subgraph
  //       Create an ExecutionFrame for each iteration that splices the input/output tensors
  //       Iterate

  NameMLValMap feeds;

  // iterate the variadic inputs
  for (int i = 0, end = num_variadic_inputs; i < end; ++i) {
    auto& input_tensor = *ctx->Input<Tensor>(i);
    auto data_type = input_tensor.DataType();
    (void)data_type;

    // the ordering of the Scan inputs should match the ordering of the subgraph inputs
    auto name = graph_inputs[i]->Name();
    feeds[name] = *ctx_impl->GetInputMLValue(i);
  }

  std::vector<MLValue> fetches;
  std::vector<std::string> output_names;

  fetches.reserve(graph_outputs.size());
  output_names.reserve(graph_outputs.size());

  for (int i = 0, end = num_variadic_outputs; i < end; ++i) {
    // TODO: Need to handle shape/type inference for subgraphs.
    // For now copy shape from subgraph output as we're only doing one iteration so far
    auto& go = graph_outputs.at(i);
    TensorShape shape{Lotus::Utils::GetTensorShapeFromTensorShapeProto(*go->Shape())};

    // make sure the output tensor is created so GetOutputMLValue will succeed
    IGNORE_RETURN_VALUE(ctx->Output(i, shape));

    // the ordering of the Scan outputs should match the ordering of the subgraph outputs
    auto name = graph_outputs[i]->Name();
    output_names.push_back(name);

    MLValue* p_mlvalue = ctx_impl->GetOutputMLValue(i);
    LOTUS_ENFORCE(p_mlvalue, "Output MLValue has not been created for output ", i);

    fetches.push_back(*p_mlvalue);
  }

  // Create Executor and run graph.
  SequentialExecutor executor;
  auto status = executor.Execute(*session_state, feeds, output_names, fetches, ctx->Logger());

  return status;
}

Status Scan::ComputeImpl() const {
  return Status::OK();
}
}  // namespace Lotus
