#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph.h"
#include "gsl/span"
#include "gsl/gsl_util"

namespace Lotus {

class SessionState;

// A very light-weight class, which works as an aggregated
// view of all data needed for constructing a Kernel instance.
// NOTE: it does not own/hold any objects.
class OpKernelInfo : public OpNodeProtoHelper<ProtoHelperNodeContext> {
 public:
  explicit OpKernelInfo(const LotusIR::Node& node,
                        const KernelDef& kernel_def,
                        const IExecutionProvider& execution_provider,
                        const SessionState& session_state);

  OpKernelInfo(const OpKernelInfo& other);

  const AllocatorInfo& GetAllocatorInfo(MemType mem_type) const;

  const KernelDef& GetKernelDef() const;

  const IExecutionProvider* GetExecutionProvider() const noexcept;

  const LotusIR::Node& node() const noexcept;

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const;

 private:
  LOTUS_DISALLOW_MOVE(OpKernelInfo);
  LOTUS_DISALLOW_ASSIGN(OpKernelInfo);

  const LotusIR::Node& node_;
  const KernelDef& kernel_def_;
  // For non cpu/cuda case, this pointer should be set so that function kernel
  // will delegate kernel compute call to <execution_provider> compute call.
  gsl::not_null<const ::Lotus::IExecutionProvider*> execution_provider_; 
  ProtoHelperNodeContext proto_helper_context_;
  const SessionState& session_state_;
};

}  // namespace Lotus
