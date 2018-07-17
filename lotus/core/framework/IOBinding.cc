#include "IOBinding.h"
#include "core/graph/graph.h"  // for LotusIR::ProviderType
#include "core/common/logging/logging.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
IOBinding::IOBinding(const SessionState& session_state) : session_state_(session_state) {
}

Common::Status IOBinding::BindInput(const std::string& name, const MLValue& ml_value) {
  if (!ml_value.IsTensor()) {
    feeds_.insert({name, ml_value});
    return Status::OK();
  }

  MLValue new_mlvalue;
  LOTUS_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state_, name, ml_value, new_mlvalue));
  feeds_.insert({name, new_mlvalue});
  return Status::OK();
}

static Common::Status AllocateHelper(const SessionState& session_state,
                                     LotusIR::ProviderType provider_type,
                                     const MLValue& fetched_mlvalue,
                                     MLValue& output_mlvalue) {
  auto* p_provider = session_state.GetExecutionProvider(provider_type);
  LOTUS_ENFORCE(p_provider);
  auto allocator = p_provider->GetAllocator();
  LOTUS_ENFORCE(allocator != nullptr);
  auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
  void* buffer = allocator->Alloc(fetched_tensor.DataType()->Size() * fetched_tensor.Shape().Size());
  LOTUS_ENFORCE(buffer);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                              fetched_tensor.Shape(),
                                                              buffer,
                                                              allocator->Info(),
                                                              allocator);
  output_mlvalue.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

// TODO should we handle the case of one input name feeding 2 nodes placed on different
// devices.
Common::Status IOBinding::CopyOneInputAcrossDevices(const SessionState& session_state,
                                                    const std::string& input_name,
                                                    const MLValue& orig_mlvalue,
                                                    MLValue& new_mlvalue) {
  std::vector<SessionState::NodeInfo> node_info_vec;
  LOTUS_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  for (auto& node_info : node_info_vec) {
    size_t index = node_info.index;
    auto& node = *node_info.p_node;
    const KernelCreateInfo* kci = node_info.kci;
    const auto* input_mem_types = (kci != nullptr) ? &kci->kernel_def->InputMemoryType() : nullptr;

    // node may declare input_mem_type to be on CPU explicitly
    bool input_on_cpu = input_mem_types && MemTypeOnCpuExplicitly(*input_mem_types, index);
    auto& required_provider_type = input_on_cpu ? LotusIR::kCpuExecutionProvider : node.GetExecutionProviderType();
    if (!orig_mlvalue.IsTensor()) {
      // copying not supported for non-tensor types
      new_mlvalue = orig_mlvalue;
      return Status::OK();
    }
    auto& input_tensor = orig_mlvalue.Get<Tensor>();
    auto& input_tensor_loc = input_tensor.Location();
    auto* p_input_provider = session_state.GetExecutionProvider(input_tensor_loc);
    if (!p_input_provider) {
      p_input_provider = session_state.GetExecutionProvider(LotusIR::kCpuExecutionProvider);
    }
    LOTUS_ENFORCE(p_input_provider);

    auto input_provider_type = p_input_provider->Type();
    if (input_provider_type == required_provider_type && input_tensor_loc.mem_type == kMemTypeDefault) {
      new_mlvalue = orig_mlvalue;
      return Status::OK();
    }

    auto* node_provider = session_state.GetExecutionProvider(required_provider_type);
    LOTUS_ENFORCE(node_provider);
    LOTUS_RETURN_IF_ERROR(AllocateHelper(session_state, required_provider_type, orig_mlvalue, new_mlvalue));
    auto* new_tensor = new_mlvalue.GetMutable<Tensor>();
    auto* node_exec_provider = session_state.GetExecutionProvider(required_provider_type);
    LOTUS_ENFORCE(node_exec_provider);

    // our CPU exec provider doesn't support copy from GPU->CPU
    if (required_provider_type != LotusIR::kCpuExecutionProvider) {
      LOTUS_RETURN_IF_ERROR(node_exec_provider->CopyTensor(input_tensor, *new_tensor));
    } else {
      LOTUS_RETURN_IF_ERROR(p_input_provider->CopyTensor(input_tensor, *new_tensor));
    }
  }

  return Status::OK();
}

static Common::Status SyncProviders(const SessionState::NameNodeInfoMapType& node_info_map,
                                    const SessionState& session_state) {
  std::set<std::string> providers;
  for (auto& pair : node_info_map) {
    for (auto& node_info : pair.second) {
      if (node_info.p_node->GetExecutionProviderType() != LotusIR::kCpuExecutionProvider) {
        providers.insert(node_info.p_node->GetExecutionProviderType());
      }
    }
  }
  for (auto& provider_type : providers) {
    auto* p_provider = session_state.GetExecutionProvider(provider_type);
    if (!p_provider) {
      continue;
    }
    LOTUS_RETURN_IF_ERROR(p_provider->Sync());
  }
  return Status::OK();
}

Common::Status IOBinding::SynchronizeInputs() {
  LOTUS_RETURN_IF_ERROR(SyncProviders(session_state_.GetInputNodeInfoMap(), session_state_));
  return Status::OK();
}

Common::Status IOBinding::SynchronizeOutputs() {
  LOTUS_RETURN_IF_ERROR(SyncProviders(session_state_.GetOutputNodeInfoMap(), session_state_));
  return Status::OK();
}

static std::pair<bool, size_t> Contains(const std::vector<std::string>& output_names, const std::string& oname) {
  auto it = std::find(std::begin(output_names), std::end(output_names), oname);
  if (it == std::end(output_names)) {
    return {false, 0};
  } else {
    return {true, it - std::begin(output_names)};
  }
}

Common::Status IOBinding::BindOutput(const std::string& name, const MLValue& ml_value) {
  auto rc = Contains(output_names_, name);
  if (rc.first) {
    outputs_[rc.second] = ml_value;
    return Status::OK();
  }

  output_names_.push_back(name);
  outputs_.push_back(ml_value);
  return Status::OK();
}

const std::vector<std::string>& IOBinding::GetOutputNames() const {
  return output_names_;
}

std::vector<MLValue>& IOBinding::GetOutputs() {
  return outputs_;
}

const std::unordered_map<std::string, MLValue>& IOBinding::GetInputs() const {
  return feeds_;
}

AllocatorPtr IOBinding::GetCPUAllocator(LotusIR::ProviderType provider_type) const {
  auto* p_provider = session_state_.GetExecutionProvider(provider_type);
  LOTUS_ENFORCE(p_provider);
  auto allocator = p_provider->GetAllocator(kMemTypeCPU);

  // if the provider does not implement CPU allocator, fall back to CPU
  if (allocator) {
    return allocator;
  } else {
    auto* cpu_provider = session_state_.GetExecutionProvider(LotusIR::kCpuExecutionProvider);
    return cpu_provider->GetAllocator();
  }
}

}  // namespace Lotus
