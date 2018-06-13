#include "transformer_memcpy.h"

namespace Lotus {

bool TransformerMemcpyImpl::ModifyGraph() {
  bool modified = false;

  // find defs that require copy
  for (auto& node : graph_->Nodes()) {
    if (graph_->IsSourceNode(node) || graph_->IsSinkNode(node))
      continue;

    if (node.GetExecutionProviderType().empty() && KernelRegistry::Instance().CanExecutionProviderCreateKernel(node, provider_)) {
      node.SetExecutionProviderType(provider_);
      modified = true;
    }

    ProcessDefs(node);
  }

  // for initializers shared by different providers, create dups
  ProcessInitializers();

  // HACKHACK: here assume graph input/output to be in CPU
  // we need to avoid copy weights to GPU for every eval (a policy for execution provider?)
  // besides, for input data, we should allow GPU copy to happen in parallel with computation
  auto& graph_inputs = graph_->GetInputs();
  non_provider_defs_.insert(graph_inputs.cbegin(), graph_inputs.cend());

  auto& graph_outputs = graph_->GetOutputs();
  non_provider_defs_.insert(graph_outputs.cbegin(), graph_outputs.cend());

  // find the defs that are both input and output for provider nodes
  // and only handle the case where non_provider_defs_ does not share items
  // with provider_inout_defs (produced and consumed by provider)
  // TODO: support that case once we have a model test for it
  bool unsupported = false;
  std::for_each(
      non_provider_defs_.begin(),
      non_provider_defs_.end(),
      [this, &unsupported](const LotusIR::NodeArg* def) {
        if (provider_input_defs_.find(def) != provider_input_defs_.end() &&
            provider_output_defs_.find(def) != provider_output_defs_.end())
          unsupported = true;
      });

  if (unsupported) {
    LOGS_DEFAULT(ERROR) << "TransformerMemCpyImpl: Unsupported graph";
  } else if (non_provider_defs_.size()) {
    for (auto p_node : provider_nodes_) {
      AddCopyNode(p_node->InputDefs(), true);
      AddCopyNode(p_node->OutputDefs(), false);
    }
    for (auto p_node : provider_nodes_) {
      p_node->ReplaceDefs(replacements_);
    }
    return true;
  }
  return modified;
}

void TransformerMemcpyImpl::ProcessDefs(LotusIR::Node& node) {
  if (node.GetExecutionProviderType() == provider_) {
    provider_nodes_.insert(&node);
    node.ForEachDef([this](const LotusIR::NodeArg* arg, bool is_input) {
      if (is_input)
        provider_input_defs_.insert(arg);
      else
        provider_output_defs_.insert(arg);
    });
  } else {
    // TODO: copy between devices? i.e. multiple GPUs
    LOTUS_ENFORCE(node.GetExecutionProviderType() == LotusIR::kCpuExecutionProvider || node.GetExecutionProviderType().empty());
    node.ForEachDef([this](const LotusIR::NodeArg* arg, bool /*is_input*/) {
      non_provider_defs_.insert(arg);
    });
  }
}

void TransformerMemcpyImpl::AddCopyNode(const ConstPointerContainer<std::vector<LotusIR::NodeArg*>>& args, bool is_input) {
  for (const gsl::not_null<const LotusIR::NodeArg*> arg : args) {
    if (!arg->Exists())
      continue;

    if (non_provider_defs_.find(&*arg) != non_provider_defs_.end() &&
        replacements_.find(arg) == replacements_.end()) {
      auto writable_arg = const_cast<LotusIR::NodeArg*>(&*arg);

      // create unique name for new def
      std::string new_def_name = graph_->GenerateNodeArgName("MemcpyDef");

      auto* new_arg = &graph_->GetOrCreateNodeArg(new_def_name, writable_arg->TypeAsProto());
      auto* src_arg = is_input ? writable_arg : new_arg;
      auto* dst_arg = is_input ? new_arg : writable_arg;

      // create unique name for copy node
      std::string new_node_name = graph_->GenerateNodeName("Memcpy");

      // only add node in this loop, editing happens outside of args iteration here
      const auto op_name = is_input ? "MemcpyFromHost" : "MemcpyToHost";
      auto new_node = graph_->AddNode(new_node_name, op_name, "Copy from/to host memory",
                                      std::vector<LotusIR::NodeArg*>{src_arg},
                                      std::vector<LotusIR::NodeArg*>{dst_arg});
      new_node->SetExecutionProviderType(provider_);
      replacements_.insert(std::make_pair(writable_arg, new_arg));
    }
  }
}

static const LotusIR::NodeArg* FindNodeArg(std::set<const LotusIR::NodeArg*> def_set, const std::string& name) {
  auto iter = std::find_if(
      def_set.begin(),
      def_set.end(),
      [name](const LotusIR::NodeArg* def) {
        return def->Name() == name;
      });
  if (iter != def_set.end())
    return *iter;
  else
    return nullptr;
}

// We duplicate any initializer that is used by both provider nodes and non-provider nodes
// to ensure that provider nodes and non-provider nodes don't share initializers, as they
// need to stay in different memory locations.
void TransformerMemcpyImpl::ProcessInitializers() {
  std::map<const LotusIR::NodeArg*, LotusIR::NodeArg*> replacements;
  for (const auto& pair : graph_->GetAllInitializedTensors()) {
    const auto& name = pair.first;
    const LotusIR::NodeArg* provider_def = FindNodeArg(provider_input_defs_, name);
    const LotusIR::NodeArg* non_provider_def = FindNodeArg(non_provider_defs_, name);
    if (provider_def != nullptr && non_provider_def != nullptr) {
      std::string new_def_name = graph_->GenerateNodeArgName(name);
      auto& new_def = graph_->GetOrCreateNodeArg(new_def_name, provider_def->TypeAsProto());

      const TensorProto* tensor_proto = nullptr;
      graph_->GetInitializedTensor(name, &tensor_proto);
      TensorProto new_tensor_proto = *tensor_proto;
      *(new_tensor_proto.mutable_name()) = new_def_name;
      graph_->AddInitializedTensor(new_tensor_proto);

      replacements.insert(std::make_pair(provider_def, &new_def));
    }
  }

  for (auto p_node : provider_nodes_) {
    p_node->ReplaceDefs(replacements);
  }
}

}  // namespace Lotus
