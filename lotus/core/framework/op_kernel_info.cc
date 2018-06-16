#include "core/framework/op_kernel_info.h"
#include "core/framework/session_state.h"

namespace Lotus {

OpKernelInfo::OpKernelInfo(const LotusIR::Node& node,
                           const KernelDef& kernel_def,
                           const IExecutionProvider* execution_provider,
                           const SessionState& session_state)
    : OpNodeProtoHelper(&proto_helper_context_),
      proto_helper_context_(node),
      node_(node),
      kernel_def_(kernel_def),
      execution_provider_(execution_provider),
      session_state_(session_state) {}

OpKernelInfo::OpKernelInfo(const OpKernelInfo& other)
    : OpKernelInfo(other.node_, other.kernel_def_, other.execution_provider_, other.session_state_) {}

const AllocatorInfo& OpKernelInfo::GetAllocatorInfo(MemType mem_type) const {
  return execution_provider_->GetAllocatorMap().at(mem_type)->Info();
}

const KernelDef& OpKernelInfo::GetKernelDef() const {
  return kernel_def_;
}

const IExecutionProvider* OpKernelInfo::GetExecutionProvider() const noexcept {
  return execution_provider_;
}

const LotusIR::Node& OpKernelInfo::node() const noexcept {
  return node_;
}

bool OpKernelInfo::TryGetConstantInput(int input_index, const Tensor** constant_input_value) const {
  if (input_index < 0 || input_index >= node_.InputDefs().size()) {
    return false;
  }
  auto& input_arg_name = node_.InputDefs()[input_index]->Name();
  int input_arg_index = -1;
  if (!session_state_.GetMLValueIdx(input_arg_name, &input_arg_index).IsOK()) {
    return false;
  }

  auto& initializers = session_state_.GetInitializedTensors();
  auto iter = initializers.find(input_arg_index);
  if (initializers.end() == iter) {
    return false;
  }
  if (!iter->second.IsTensor()) {
    // Only constant Tensor input is support right now, since we're using initializers to store the data.
    return false;
  }
  *constant_input_value = &iter->second.Get<Tensor>();
  return true;
}
}  // namespace Lotus
