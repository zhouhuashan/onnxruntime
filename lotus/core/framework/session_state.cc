#include "core/framework/session_state.h"

#include <sstream>

#include "core/common/logging.h"

namespace Lotus {

void SessionState::SetGraph(const LotusIR::Graph* graph) {
  p_graph_ = graph;
  session_kernels_.resize(p_graph_->NumberOfNodes());
}

const LotusIR::Graph* SessionState::GetGraph() const {
  return p_graph_;
}

const std::vector<unique_ptr<OpKernel>>& SessionState::GetKernelVector() const {
  return session_kernels_;
}

const OpKernel* SessionState::GetKernel(LotusIR::NodeIndex node_id) const {
  if (node_id >= session_kernels_.size()) {
    return nullptr;
  }
  
  return session_kernels_[node_id].get();
}

void SessionState::AddKernel(LotusIR::NodeIndex nodeId, std::unique_ptr<OpKernel> p_kernel) {
  // assumes vector is already resize()'ed to the number of nodes in the graph
  // and the nodeIds space is dense
  LOTUS_ENFORCE(p_graph_ && (session_kernels_.size() == p_graph_->NumberOfNodes()));
  session_kernels_[nodeId] = std::move(p_kernel);
}

void SessionState::AddExecutionProvider(const std::string& provider_id, std::unique_ptr<IExecutionProvider> p_exec_provider) {
  exec_provider_set_.provider_idx_map.insert({provider_id, exec_provider_set_.exec_providers.size()});
  exec_provider_set_.exec_providers.push_back(std::move(p_exec_provider));
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

void SessionState::SetExecutionPlan(std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan) {
  p_seq_exec_plan_ = std::move(p_seq_exec_plan);
}

const SequentialExecutionPlan* SessionState::GetExecutionPlan() const {
  return p_seq_exec_plan_.get();
}

void SessionState::AddMLValueNameIdx(const std::string& name, int idx) {
  int idx_ret;
  Common::Status status = GetMLValueIdx(name, &idx_ret);
  if (status.IsOK()) {
    LOG(WARNING) << "MLValue name " << name << " already exists in the MLValueName -> idx map";
    return;
  }

  if (idx > mlvalue_max_idx_) {
    mlvalue_max_idx_ = idx;
  }
  mlvalue_name_idx_map_.insert({name, idx});
}

// returns OK() if value is found
Common::Status SessionState::GetMLValueIdx(const std::string& name, int* idx) const {
  auto it = mlvalue_name_idx_map_.find(name);
  if (it == mlvalue_name_idx_map_.end()) {
    std::ostringstream ostr;
    ostr << "Could not find MLValue with name: " << name;
    return Common::Status(Common::LOTUS, Common::FAIL, ostr.str());
  }

  *idx = it->second;
  return Common::Status::OK();
}

size_t SessionState::GetNumMLValues() const {
  return mlvalue_name_idx_map_.size();
}

int SessionState::GetMaxMLValueIdx() const {
  return mlvalue_max_idx_;
}

void SessionState::AddInitializedTensor(int mlvalue_index, const MLValue& mlvalue) {
  LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index <= GetMaxMLValueIdx());
  initialized_tensors_.insert({mlvalue_index, mlvalue});
}

const std::unordered_map<int, MLValue>& SessionState::GetInitializedTensors() const {
  return initialized_tensors_;
}

}  // namespace Lotus
