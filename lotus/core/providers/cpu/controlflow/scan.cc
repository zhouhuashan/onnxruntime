#include "core/providers/cpu/controlflow/scan.h"

#include "core/framework/op_kernel_context_impl.h"
#include "core/framework/session_state.h"

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
  auto ctx_impl = static_cast<OpKernelContextImpl*>(ctx);
  auto* session_state = ctx_impl->SubgraphSessionState("body");
  LOTUS_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");

  auto num_inputs = ctx->InputCount();
  auto num_outputs = ctx->OutputCount();

  LOTUS_ENFORCE(num_inputs > 0 && num_outputs > 0);

  // temporary hack to copy input to output so the operator 'works' end to end
  auto X = ctx->Input<Tensor>(0);
  auto Y = ctx->Output(0, X->Shape());

  auto input = X->DataAsSpan<float>();
  auto data = Y->MutableDataAsSpan<float>();

  std::copy(input.cbegin(), input.cend(), data.begin());

  // TODO:
  // Create Executor and run graph. Need to handle slicing input for each iteration.
  // auto& exec_plan = *session_state_->GetExecutionPlan();

  return Status::OK();
}
}  // namespace Lotus
