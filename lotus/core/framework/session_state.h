#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "core/framework/mem_pattern.h"

namespace Lotus {
// SessionState should be modified by the inference session class only.
// It is supposed to be passed by const-ref only to all the executors.
class SessionState {
 public:
  SessionState() = default;

  SessionState(int num_nodes) : session_kernels_(num_nodes) {
    // TODO Dummy constructor for now to add a basic test.
  }

  // graph
  void SetGraph(const LotusIR::Graph* graph);
  const LotusIR::Graph* GetGraph() const;

  // kernels
  void SetKernelVectorSize(size_t size);
  const OpKernel* GetKernel(LotusIR::NodeIndex node_id) const;
  void AddKernel(LotusIR::NodeIndex node_id, std::unique_ptr<OpKernel> p_kernel);
  const std::vector<unique_ptr<OpKernel>>& GetKernelVector() const;

  // exec providers
  IExecutionProvider* GetExecutionProvider(const std::string& provider_id) const;
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
  Get cached memory pattern based on input shapes
  */
  const MemoryPatternGroup* GetMemoryPatternGroup(const std::vector<TensorShape>& input_shapes) const;

  /**
  Set generated memory pattern with a given input shapes
  */
  Status SetMemoryPatternGroup(const std::vector<TensorShape>& input_shape, std::unique_ptr<MemoryPatternGroup> mem_patterns);

  /**
  Set enable memory pattern flag
  */
  void SetEnableMemoryPattern(bool flag);

  /**
  Get enable memory pattern flag
  */
  bool GetEnableMemoryPattern() const;

 private:
  // cache of the constructed kernels to avoid spending construction
  // time per executor
  std::vector<unique_ptr<OpKernel>> session_kernels_;
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

  // switch for enable memory pattern optimization or not.
  bool enable_mem_pattern_;
  // lock for the mem_patterns_
  mutable std::mutex mem_patterns_lock_;
  // cache for the generated mem_patterns. key is cauclated based on input shapes.
  std::map<int64_t, std::unique_ptr<MemoryPatternGroup>> mem_patterns_;

  // TODO add more
};
}  // namespace Lotus
