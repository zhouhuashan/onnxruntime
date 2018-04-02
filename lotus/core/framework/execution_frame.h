#pragma once

#include <mutex>
#include <vector>
#include "core/common/status.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
// #include "core/framework/session_state.h"

namespace Lotus {

class SessionState;

class ExecutionFrame {
 public:
  typedef MLValue* NodeArgValue;

  // For arena management design, we could have two options:
  // 1. For each device, arena is global in the entire Process,
  //    like all the infer session shared the same cpu arena.
  //    The benefit is it gives us protential to share memory
  //    between different session/request, but will pay the cost
  //    for locking in concurrency.
  // 2. Each executor will host its own arena, after memory planning
  //    we could allocate more efficient with no locking. But the
  //    peak working set memory usage will equal to arenas used by
  //    all concurrent requests. And we might need Arena to have
  //    different strategy for different graph to address it.
  // No matter which approach we chose, we definitly need to hold
  // Arena in execution frame, the question is should arena owned
  // by execution frame or not. That's why we make this typedef here.
  // Right now the milestone1 implementation goes with option 1.
  // So I make it naked ptr here. Once we finished option 2 and got
  // better result, we can replace this part with something like unique_ptr.
  typedef IArenaAllocator* ArenaPtr;

  ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                 const std::vector<std::string>& output_names,
                 const SessionState& session_state);

  ~ExecutionFrame() {
  }

  Status AllocateMLValueTensorSelfOwnBuffer(int mlvalue_index,
                                            const MLDataType element_type,
                                            const AllocatorInfo& location,
                                            const TensorShape& shape);

  Status AllocateMLValueTensorPreAllocateBuffer(int mlvalue_index_to_allocate,
                                                int mlvalue_index_reuse,
                                                const MLDataType element_type,
                                                const AllocatorInfo& location,
                                                const TensorShape& shape);

  // ?? Cheng: What about non-tensor values??
  // ?? Cheng: There are cases we may not want to use LOTUS_ENFORCE??
  // ?? Cheng: Graph must be immutable for GetNodesInTopologicalOrder??
  // Create tensor at index mlvalue, and allocate buffer for it.
  // This tensor will own this buffer.
  // This method is not thread safe!
  Status AllocateTensorWithSelfOwnBuffer(const int index,
                                         const MLDataType element_type,
                                         const AllocatorInfo& location,
                                         const TensorShape& shape);

  // Create tensor at index mlvalue, with pre-allocate buffer
  // This tensor does not own the buffer.
  // The executor / planner need to be careful about the
  // lifetime of the buffer. Tensor itself won't manage it.
  // This method is not thread safe!
  Status AllocateTensorWithPreAllocateBuffer(const int offset,
                                             void* pBuffer,
                                             const MLDataType element_type,
                                             const AllocatorInfo& location,
                                             const TensorShape& shape);

  const MLValue& GetMLValue(int mlvalue_index) const {
    LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index < all_values_.size());
    return all_values_[mlvalue_index];
  }

  // Index to the first argument of the given node.
  int GetFirstArgIndex(LotusIR::NodeIndex index) {
    LOTUS_ENFORCE(index >= 0 && index < node_offsets_.size());
    return node_offsets_[index];
  }

  template <typename T>
  const T* GetValue(int index) const {
    LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
    return &all_values_[node_values_[index]].Get<T>();
  }

  template <typename T>
  T* GetMutableValue(int index) {
    LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
    return all_values_[node_values_[index]].GetMutable<T>();
  }

  ArenaPtr GetArena(const AllocatorInfo& info) {
    for (auto arena : arenas_) {
      if (arena->Info() == info)
        return arena;
    }
    return nullptr;
  }

  void ReleaseMLValue(int mlvalue_idx) {
    LOTUS_ENFORCE(mlvalue_idx >= 0 || mlvalue_idx < all_values_.size());
    all_values_[mlvalue_idx] = MLValue();
  }

  const Lotus::SessionState& SessionState() const {
    return session_state_;
  }

 private:
  friend class OpKernelContext;

  // The TestUtils need hack this class to provide input/output
  // tensors since the class is not fully implemented yet.
  friend class Lotus::Test::TestUtils;

  // This method is not thread safe!
  void Release(const int offset);

  Common::Status AllocateAsPerAllocationPlan(int mlvalue_index,
                                             const TensorShape& shape);

  Status AllocateMLValueTensorSelfOwnBuffer(MLValue* p_mlvalue,
                                            const MLDataType element_type,
                                            const AllocatorInfo& location,
                                            const TensorShape& shape);

  void Init(const LotusIR::Graph* graph,
            const std::unordered_map<string, MLValue>& feeds,
            const std::vector<string>& outputs);

  void SetupNodeArg(LotusIR::NodeArg* arg);

  Status AllocateTensorWithPreAllocateBufferHelper(MLValue* p_mlvalue,
                                                   void* pBuffer,
                                                   const MLDataType element_type,
                                                   const AllocatorInfo& location,
                                                   const TensorShape& shape);

  // This method is not thread safe!
  Tensor* GetOrCreateTensor(int index, const TensorShape& shape);

  void InitArenas() {
    // For milestone 1, we only have CPU arena.
    // If later we want executor to host its own arena
    // Need to update this part.
    auto& alloc_mgr = AllocatorManager::Instance();
    arenas_.push_back(&alloc_mgr.GetArena(CPU));
  }

  std::mutex mutex_;
  Status status_;

  // The values for the inputs and outputs of the nodes.
  // This vector contains the indices into the all_values_ vector.
  std::vector<int> node_values_;

  // All the intermediate values for the entire graph.
  // Input and Output values are passed in by executors
  std::vector<MLValue> all_values_;

  // The start index into node_values_ for all the nodes.
  std::vector<int> node_offsets_;

  // i-th kernel is still waiting for pending_counts_[i] inputs.
  std::vector<int> pending_counts_;  // not used currently

  // The arenas used for current execution
  // Like mentioned in comments above, we could have two approach:
  // Arena owned by global allocator manager, or arena owned by
  // Execution frame. Currently we are implment by global arena approach
  // So here is a list of raw pointer and execution frame don't need
  // release them. If we switch to another approach later, we should
  // define ArenaPtr as unique_ptr here.
  vector<ArenaPtr> arenas_;

  std::unordered_map<string, int> value_name_to_index_;

  const Lotus::SessionState& session_state_;
};
}  // namespace Lotus
