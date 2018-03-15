#include "core/framework/session_state.h"

namespace Lotus {

void SessionState::Init(const LotusIR::Graph* graph) {
  p_graph_ = graph;
  session_kernels_.resize(p_graph_->NumberOfNodes());
}

const LotusIR::Graph* SessionState::GetGraph() const {
  return p_graph_;
}

const std::vector<unique_ptr<OpKernel>>& SessionState::GetKernelVector() const {
  return session_kernels_;
}

OpKernel* SessionState::GetKernel(LotusIR::NODEINDEX node_id) const {
  if (node_id >= session_kernels_.size()) {
    return nullptr;
  }
  return session_kernels_[node_id].get();
}

void SessionState::AddKernel(LotusIR::NODEINDEX nodeId, std::unique_ptr<OpKernel> p_kernel) {
  // assumes vector is already resize()'ed to the number of nodes in the graph
  // and the nodeIds space is dense
  LOTUS_ENFORCE(session_kernels_.size() == p_graph_->NumberOfNodes());
  session_kernels_[nodeId] = std::move(p_kernel);
}

void SessionState::AddExecutionProvider(const std::string& provider_id, std::unique_ptr<IExecutionProvider> exec_provider) {
  exec_provider_set_.provider_idx_map.insert(std::make_pair(provider_id, exec_provider_set_.exec_providers.size()));        
  exec_provider_set_.exec_providers.push_back(std::move(exec_provider));  
}

IExecutionProvider* SessionState::GetExecutionProvider(const std::string& provider_id) const {
  auto it = exec_provider_set_.provider_idx_map.find(provider_id);
  if (it == exec_provider_set_.provider_idx_map.end()) {
    return nullptr;
  }
  
  LOTUS_ENFORCE(it->second < exec_provider_set_.exec_providers.size());
  return exec_provider_set_.exec_providers[it->second].get();  
}

const std::vector<std::unique_ptr<IExecutionProvider>>& SessionState::GetExecutionProviders() const {
  return exec_provider_set_.exec_providers;
}

}
