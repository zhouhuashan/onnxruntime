#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_provider.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"
#include "core/graph/graph.h"

namespace Lotus {
class OpKernel;
class KernelDef;
struct SequentialExecutionPlan;
struct MemoryPatternGroup;
// SessionState should be modified by the inference session class only.
// It is supposed to be passed by const-ref only to all the executors.
class SessionState {
 public:
  SessionState() = default;

  // graph
  void SetGraph(const LotusIR::Graph* graph);
  const LotusIR::Graph* GetGraph() const;

  // kernels
  // Get kernel for specified node.
  // It should called right before graph execution only.
  const OpKernel* GetKernel(LotusIR::NodeIndex node_id) const;
  const KernelDef* GetKernelDef(LotusIR::NodeIndex node_id) const;
  const AllocatorInfo& GetAllocatorInfo(LotusIR::NodeIndex node_id, MemType mem_type) const;
  void AddKernel(LotusIR::NodeIndex node_id, std::unique_ptr<OpKernel> p_kernel);

  // exec providers
  IExecutionProvider* GetExecutionProvider(LotusIR::ProviderType provider_id) const;

  IExecutionProvider* GetExecutionProvider(const AllocatorInfo& allocator_info) const;

  void AddExecutionProvider(const std::string& provider_id,
                            std::unique_ptr<IExecutionProvider> exec_provider);
  const std::vector<std::unique_ptr<IExecutionProvider>>& GetExecutionProviders() const;
  // return nullptr if the allocator not found
  AllocatorPtr GetAllocator(const AllocatorInfo& allocator_info) const;

  // MLValueName idx map
  void AddMLValueNameIdx(const std::string& name, int idx);
  Common::Status GetMLValueIdx(const std::string& name, int* idx) const;
  size_t GetNumMLValues() const;
  int GetMaxMLValueIdx() const;
  const std::unordered_map<std::string, int>& GetMLValueIdxMap() const;

  // initialized tensors
  /**
  * Adds an initialized tensor (weight) so that it can be used by the
  * execution frame to setup the appropriate MLValue vectors.
  */
  void AddInitializedTensor(int mlvalue_index, const MLValue& mlvalue);

  /**
  * Gets the list of all initialized tensors (weights) so that it can be used by the
  * execution frame to setup the appropriate MLValue vectors.
  */
  const std::unordered_map<int, MLValue>& GetInitializedTensors() const;

  // execution plan
  void SetExecutionPlan(std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan);
  const SequentialExecutionPlan* GetExecutionPlan() const;

  /**
  Set the logger to use for this session. 
  */
  SessionState& SetLogger(const Logging::Logger& logger);

  /**
  Get the logger for this session. 
  Falls back to returning Logging::LoggingManager::DefaultLogger if SetLogger has not been called.
  */
  const Logging::Logger& Logger() const;

  /**
  Set the profiler for this session.
  */
  void SetProfiler(Profiling::Profiler& profiler);

  /**
  Get the profiler for this session. It needs to be enabled via the InferenceSession to perform
  profiling actions.
  */
  Profiling::Profiler& Profiler() const;

  /**
  Get cached memory pattern based on input shapes
  */
  const MemoryPatternGroup* GetMemoryPatternGroup(const std::vector<TensorShape>& input_shapes) const;

  /**
  Set generated memory pattern with a given input shapes. 
  Const as it's an internal cache update only.
  */
  Status UpdateMemoryPatternGroupCache(const std::vector<TensorShape>& input_shape,
                                       std::unique_ptr<MemoryPatternGroup> mem_patterns) const;

  /**
  Set enable memory pattern flag
  */
  void SetEnableMemoryPattern(bool flag);

  /**
  Get enable memory pattern flag
  */
  bool GetEnableMemoryPattern() const;

  const KernelRegistryManager& GetKernelRegistryManager() const;
  KernelRegistryManager& GetKernelRegistryManager();

  struct NodeInfo {
    NodeInfo(size_t index0, const LotusIR::Node* p_node0, const KernelCreateInfo* kci0)
        : index(index0),
          p_node(p_node0),
          kci(kci0) {
    }
    NodeInfo() = default;

    size_t index;
    const LotusIR::Node* p_node = nullptr;
    const KernelCreateInfo* kci = nullptr;
  };
  using NameNodeInfoMapType = std::unordered_map<std::string, std::vector<NodeInfo>>;
  void AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info);
  Common::Status GetInputNodeInfo(const std::string& input_name, std::vector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetInputNodeInfoMap() const;

  void AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info);
  const NameNodeInfoMapType& GetOutputNodeInfoMap() const;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(SessionState);

  // cache of the constructed kernels to avoid spending construction
  // time per executor
  std::unordered_map<LotusIR::NodeIndex, std::unique_ptr<OpKernel>> session_kernels_;
  const LotusIR::Graph* p_graph_ = nullptr;  // owned by the Model inside an InferenceSession

  struct ExecutionProviderSet {
    std::vector<std::unique_ptr<IExecutionProvider>> exec_providers;
    std::unordered_map<std::string, size_t> provider_idx_map;  // for fast lookup
    std::map<AllocatorInfo, size_t> allocator_idx_map;
  };

  ExecutionProviderSet exec_provider_set_;
  std::unordered_map<std::string, int> mlvalue_name_idx_map_;
  int mlvalue_max_idx_ = 0;

  // initialized tensorset
  std::unordered_map<int, MLValue> initialized_tensors_;  // key is mlvalue_index
  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan_ = nullptr;

  const Logging::Logger* logger_;
  Profiling::Profiler* profiler_;

  // switch for enable memory pattern optimization or not.
  bool enable_mem_pattern_ = true;
  // lock for the mem_patterns_
  mutable std::mutex mem_patterns_lock_;
  // cache for the generated mem_patterns. key is calculated based on input shapes.
  mutable std::map<int64_t, std::unique_ptr<MemoryPatternGroup>> mem_patterns_;

  // <custom_registry_manager_> contains 2 kinds of kernel registries
  // with priority from high to low as below,
  // 1. Custom execution provider type specific kernel registries.
  // 2. Common execution provider type specific kernel registries.
  // The 1st and 2nd ones are shared across sessions.
  KernelRegistryManager custom_registry_manager_;

  NameNodeInfoMapType input_names_to_nodeinfo_mapping_;
  NameNodeInfoMapType output_names_to_nodeinfo_mapping_;
};
}  // namespace Lotus
