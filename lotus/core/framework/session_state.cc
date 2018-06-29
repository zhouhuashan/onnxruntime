#include "core/framework/session_state.h"

#include <sstream>

#include "core/common/logging/logging.h"

namespace Lotus {

void SessionState::SetGraph(const LotusIR::Graph* graph) {
  p_graph_ = graph;
}

const LotusIR::Graph* SessionState::GetGraph() const {
  return p_graph_;
}

const OpKernel* SessionState::GetKernel(LotusIR::NodeIndex node_id) const {
  if (session_kernels_.count(node_id) == 0) {
    return nullptr;
  }

  return session_kernels_.find(node_id)->second.get();
}

const KernelDef* SessionState::GetKernelDef(LotusIR::NodeIndex node_id) const {
  auto node = p_graph_->GetNode(node_id);
  LOTUS_ENFORCE(nullptr != node);

  const KernelRegistry::KernelCreateInfo* kernel_create_info = nullptr;
  if (custom_registry_manager_.SearchKernelRegistry(*node, &kernel_create_info).IsOK()) {
    return kernel_create_info->kernel_def.get();
  }

  if (KernelRegistry::Instance().SearchKernelRegistry(*node, &kernel_create_info).IsOK()) {
    return kernel_create_info->kernel_def.get();
  }
  return nullptr;
}

const AllocatorInfo& SessionState::GetAllocatorInfo(LotusIR::NodeIndex node_id, MemType mem_type) const {
  auto node = p_graph_->GetNode(node_id);
  LOTUS_ENFORCE(nullptr != node);
  auto iter = exec_provider_set_.provider_idx_map.find(node->GetExecutionProviderType());
  LOTUS_ENFORCE(exec_provider_set_.provider_idx_map.end() != iter);
  auto allocator = exec_provider_set_.exec_providers[iter->second]->GetAllocatorMap().at(mem_type);
  LOTUS_ENFORCE(nullptr != allocator);
  return allocator->Info();
}

const CustomRegistryManager& SessionState::GetCustomRegistryManager() const {
  return custom_registry_manager_;
}
CustomRegistryManager& SessionState::GetCustomRegistryManager() {
  return custom_registry_manager_;
}

void SessionState::AddKernel(LotusIR::NodeIndex node_id, std::unique_ptr<OpKernel> p_kernel) {
  // assumes vector is already resize()'ed to the number of nodes in the graph
  session_kernels_[node_id] = std::move(p_kernel);
}

void SessionState::AddExecutionProvider(const std::string& provider_id,
                                        std::unique_ptr<IExecutionProvider> p_exec_provider) {
  exec_provider_set_.provider_idx_map.insert(
      {provider_id, exec_provider_set_.exec_providers.size()});
  const auto& allocator_map = p_exec_provider->GetAllocatorMap();
  for (const auto& pair : allocator_map) {
    auto allocator = pair.second;
    if (exec_provider_set_.allocator_idx_map.find(allocator->Info()) !=
        exec_provider_set_.allocator_idx_map.end()) {
      LOGS_DEFAULT(WARNING) << "Execution Provider's allocator with info:" << allocator->Info().name << ", id: " << allocator->Info().id << ", type:" << allocator->Info().type << " already register in session state";
    } else {
      exec_provider_set_.allocator_idx_map.insert(
          {allocator->Info(), exec_provider_set_.exec_providers.size()});
    }
  }
  exec_provider_set_.exec_providers.push_back(std::move(p_exec_provider));
}

IExecutionProvider* SessionState::GetExecutionProvider(LotusIR::ProviderType provider_id) const {
  auto it = exec_provider_set_.provider_idx_map.find(provider_id);
  if (it == exec_provider_set_.provider_idx_map.end()) {
    return nullptr;
  }

  LOTUS_ENFORCE(it->second < exec_provider_set_.exec_providers.size());
  return exec_provider_set_.exec_providers[it->second].get();
}

IExecutionProvider* SessionState::GetExecutionProvider(const AllocatorInfo& allocator_info) const {
  auto it = exec_provider_set_.allocator_idx_map.find(allocator_info);
  if (it == exec_provider_set_.allocator_idx_map.end()) {
    return nullptr;
  }

  LOTUS_ENFORCE(it->second < exec_provider_set_.exec_providers.size());
  return exec_provider_set_.exec_providers[it->second].get();
}

AllocatorPtr SessionState::GetAllocator(const AllocatorInfo& allocator_info) const {
  auto exec_provider = GetExecutionProvider(allocator_info);
  if (exec_provider == nullptr) {
    return nullptr;
  }

  return exec_provider->GetAllocator(allocator_info.mem_type);
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

const std::unordered_map<std::string, int>& SessionState::GetMLValueIdxMap() const {
  return mlvalue_name_idx_map_;
}

void SessionState::AddInitializedTensor(int mlvalue_index, const MLValue& mlvalue) {
  LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index <= GetMaxMLValueIdx());
  initialized_tensors_.insert({mlvalue_index, mlvalue});
}

const std::unordered_map<int, MLValue>& SessionState::GetInitializedTensors() const {
  return initialized_tensors_;
}

SessionState& SessionState::SetLogger(const Logging::Logger& logger) {
  logger_ = &logger;
  return *this;
}

const Logging::Logger& SessionState::Logger() const {
  // DefaultLogger either throws or returns a valid logger.
  const Logging::Logger* logger = logger_ != nullptr ? logger_ : &Logging::LoggingManager::DefaultLogger();
  return *logger;
}

void SessionState::SetProfiler(Profiling::Profiler& profiler) {
  profiler_ = &profiler;
}

Lotus::Profiling::Profiler& SessionState::Profiler() const {
  return *profiler_;
}

static int64_t CalculateMemoryPatternsKey(const std::vector<TensorShape>& shapes) {
  int64_t key = 0;
  for (auto& shape : shapes) {
    for (auto dim : shape.GetDims())
      key ^= dim;
  }
  return key;
}

const MemoryPatternGroup* SessionState::GetMemoryPatternGroup(const std::vector<TensorShape>& input_shapes) const {
  std::lock_guard<std::mutex> lock(mem_patterns_lock_);
  int64_t key = CalculateMemoryPatternsKey(input_shapes);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end())
    return nullptr;
  else
    return it->second.get();
}

Status SessionState::UpdateMemoryPatternGroupCache(const std::vector<TensorShape>& input_shape,
                                                   std::unique_ptr<MemoryPatternGroup> mem_patterns) const {
  int64_t key = CalculateMemoryPatternsKey(input_shape);

  std::lock_guard<std::mutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
    mem_patterns_[key] = std::move(mem_patterns);
  }

  return Status::OK();
}

void SessionState::SetEnableMemoryPattern(bool flag) {
  enable_mem_pattern_ = flag;
}

bool SessionState::GetEnableMemoryPattern() const {
  return enable_mem_pattern_;
}

void SessionState::AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info) {
  input_names_to_nodeinfo_mapping_[input_name].push_back(node_info);
}

Common::Status SessionState::GetInputNodeInfo(const std::string& input_name, std::vector<NodeInfo>& node_info_vec) const {
  if (!input_names_to_nodeinfo_mapping_.count(input_name)) {
    return Status(LOTUS, FAIL, "Failed to find input name in the mapping");
  }
  node_info_vec = input_names_to_nodeinfo_mapping_.at(input_name);
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetInputNodeInfoMap() const {
  return input_names_to_nodeinfo_mapping_;
}

void SessionState::AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info) {
  output_names_to_nodeinfo_mapping_[output_name].push_back(node_info);
}

const SessionState::NameNodeInfoMapType& SessionState::GetOutputNodeInfoMap() const {
  return output_names_to_nodeinfo_mapping_;
}

}  // namespace Lotus
