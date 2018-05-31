#include "transformer_memcpy.h"

namespace Lotus {

bool TransformerMemcpyImpl::ModifyGraph() {
  bool modified = false;

  // find unassigned constant nodes, and only make them provider constant node when outputs are provider node defs
  std::set<LotusIR::Node*> constant_nodes;
  for (auto& node : graph_->Nodes()) {
    if (graph_->IsSourceNode(node) || graph_->IsSinkNode(node))
      continue;

    if (node.GetExecutionProviderType().empty() && node.OpType() == "Constant")
      constant_nodes.insert(&node);
  }

  // find defs that require copy
  for (auto& node : graph_->Nodes()) {
    if (graph_->IsSourceNode(node) || graph_->IsSinkNode(node) || node.OpType() == "Constant")
      continue;

    if (node.GetExecutionProviderType().empty() && KernelRegistry::Instance().CanExecutionProviderCreateKernel(node, provider_)) {
      node.SetExecutionProviderType(provider_);
      modified = true;
    }

    ProcessDefs(node);
  }

  // set constants node to provider_ when used by provider nodes
  // note that no copy is needed for constant node
  for (auto c : constant_nodes) {
    auto const_out_def = c->OutputDefs()[0];
    if (std::find(provider_input_defs_.begin(), provider_input_defs_.end(), const_out_def) != provider_input_defs_.end()) {
      c->SetExecutionProviderType(provider_);
      modified = true;
    }
  }

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
      const_cast<LotusIR::Node*>(p_node)->ReplaceDefs(replacements_);
    }
    return true;
  }
  return modified;
}

void TransformerMemcpyImpl::ProcessDefs(const LotusIR::Node& node) {
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

    if (non_provider_defs_.find(&*arg) != non_provider_defs_.end()) {
      auto writable_arg = const_cast<LotusIR::NodeArg*>(&*arg);

      int id = copy_node_count_++;

      // create unique name for new def
      char str[32];
      snprintf(str, 32, "MemcpyDef_%d", id);

      auto* new_arg = &graph_->CreateOwnedNodeArg(str, writable_arg->TypeAsProto());
      auto* src_arg = is_input ? writable_arg : new_arg;
      auto* dst_arg = is_input ? new_arg : writable_arg;

      // create unique name for copy node
      snprintf(str, 32, "Memcpy_%d", id);

      // only add node in this loop, editing happens outside of args iteration here
      const auto op_name = is_input ? "MemcpyFromHost" : "MemcpyToHost";
      auto new_node = graph_->AddNode(str, op_name, "Copy from/to host memory",
                                      std::vector<LotusIR::NodeArg*>{src_arg},
                                      std::vector<LotusIR::NodeArg*>{dst_arg});
      new_node->SetExecutionProviderType(provider_);
      replacements_.insert(std::make_pair(writable_arg, new_arg));
    }
  }
}

}  // namespace Lotus
