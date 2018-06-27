#pragma once

#include <mutex>
#include <vector>
#include "core/common/status.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/framework/ml_value_patterns_planner.h"
#include "core/common/logging/logging.h"
// #include "core/framework/session_state.h"

namespace Lotus {

class SessionState;
struct MemoryPatternGroup;

struct MLValueAllocationParameters {
  TensorShape tensor_shape;
  //todo: is there any parameter needed for ml types?
};

template <typename T>
inline void VerifyShape(const T* value,
                        const MLValue* p_mlvalue,
                        const MLValueAllocationParameters& parameters) {
  if (p_mlvalue->IsTensor()) {
    const Tensor* tensor = static_cast<const Tensor*>(value);
    if (tensor->Shape() != parameters.tensor_shape) {
      LOTUS_THROW("MLValue shape verification failed.");
    }
  }
}

template <>
inline void VerifyShape<VectorMapStringToFloat>(const VectorMapStringToFloat* value,
                                                const MLValue* p_mlvalue,
                                                const MLValueAllocationParameters& parameters) {
  // no verification needed in this case.
  UNUSED_PARAMETER(value);
  UNUSED_PARAMETER(p_mlvalue);
  UNUSED_PARAMETER(parameters);
}

template <>
inline void VerifyShape<VectorMapInt64ToFloat>(const VectorMapInt64ToFloat* value,
                                               const MLValue* p_mlvalue,
                                               const MLValueAllocationParameters& parameters) {
  // no verification needed in this case.
  UNUSED_PARAMETER(value);
  UNUSED_PARAMETER(p_mlvalue);
  UNUSED_PARAMETER(parameters);
}

class ExecutionFrame {
 public:
  using NodeArgValue = MLValue*;

  ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                 const std::vector<std::string>& output_names,
                 const std::vector<MLValue>& fetches,
                 const SessionState& session_state);

  ~ExecutionFrame() = default;

  Status AllocateMLValueTensorSelfOwnBuffer(int mlvalue_index,
                                            MLDataType element_type,
                                            const AllocatorInfo& location,
                                            const TensorShape& shape,
                                            bool create_fence = false);

  Status AllocateMLValueTensorPreAllocateBuffer(int mlvalue_index_to_allocate,
                                                int mlvalue_index_reuse,
                                                MLDataType element_type,
                                                const AllocatorInfo& location,
                                                const TensorShape& shape);

  // ?? Cheng: What about non-tensor values??
  // ?? Cheng: There are cases we may not want to use LOTUS_ENFORCE??
  // ?? Cheng: Graph must be immutable for GetNodesInTopologicalOrder??
  // Create tensor at index mlvalue, and allocate buffer for it.
  // This tensor will own this buffer.
  // This method is not thread safe!
  Status AllocateTensorWithSelfOwnBuffer(int index,
                                         MLDataType element_type,
                                         const AllocatorInfo& location,
                                         const TensorShape& shape,
                                         bool create_fence = false);

  // Create tensor at index mlvalue, with pre-allocate buffer
  // This tensor does not own the buffer.
  // The executor / planner need to be careful about the
  // lifetime of the buffer. Tensor itself won't manage it.
  // This method is not thread safe!
  Status AllocateTensorWithPreAllocateBuffer(int offset,
                                             void* pBuffer,
                                             MLDataType element_type,
                                             const AllocatorInfo& location,
                                             const TensorShape& shape);

  const MLValue& GetMLValue(int mlvalue_index) const {
    LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index < all_values_.size());
    return all_values_[mlvalue_index];
  }

  // Index to the first argument of the given node.
  int GetFirstArgIndex(LotusIR::NodeIndex index) {
    LOTUS_ENFORCE(index < node_offsets_.size());
    return node_offsets_[index];
  }

  // Return nullptr if index map to an value that is an unused optional input/output
  template <typename T>
  const T* GetValue(int index) const {
    LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
    return node_values_[index] >= 0 ? &all_values_[node_values_[index]].Get<T>() : nullptr;
  }

  Fence_t GetFence(int index) const {
    LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
    return node_values_[index] >= 0 ? all_values_[node_values_[index]].Fence() : nullptr;
  }

  // Return nullptr if index map to an value that is an unused optional input/output
  MLDataType GetType(int index) const {
    LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
    return node_values_[index] >= 0 ? all_values_[node_values_[index]].Type() : nullptr;
  }

  // Return nullptr if index map to an value that is an unused optional input/output
  template <typename T>
  T* GetMutableValue(int index) {
    LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
    return node_values_[index] >= 0 ? all_values_[node_values_[index]].GetMutable<T>() : nullptr;
  }

  AllocatorPtr GetAllocator(const AllocatorInfo& info);

  void ReleaseMLValue(int mlvalue_idx) {
    LOTUS_ENFORCE(mlvalue_idx >= 0 || mlvalue_idx < all_values_.size());
    all_values_[mlvalue_idx] = MLValue();
    TraceFree(mlvalue_idx);
  }

  const Lotus::SessionState& SessionState() const {
    return session_state_;
  }

  Status GeneratePatterns(MemoryPatternGroup* out) const;

  bool HasPlan() const {
    return planner_ != nullptr;
  }

 private:
  friend class OpKernelContext;
  // This method is not thread safe!
  void Release(int offset);

  Common::Status AllocateAsPerAllocationPlan(int mlvalue_index,
                                             const MLValueAllocationParameters& parameters);

  Status AllocateMLValueTensorSelfOwnBufferHelper(int mlvalue_index,
                                                  MLDataType element_type,
                                                  const AllocatorInfo& location,
                                                  const TensorShape& shape,
                                                  bool create_fence);

  void Init(const LotusIR::Graph* graph,
            const std::unordered_map<string, MLValue>& feeds,
            const std::vector<string>& output_names,
            const std::vector<MLValue>& fetches);

  void SetupNodeArg(const LotusIR::NodeArg* arg);

  Status AllocateTensorWithPreAllocateBufferHelper(MLValue* p_mlvalue,
                                                   void* pBuffer,
                                                   MLDataType element_type,
                                                   const AllocatorInfo& location,
                                                   const TensorShape& shape);

  // This method is not thread safe!
  // Return S_OK and nullptr if index map to an value that is an unused optional input/output
  template <typename T>
  Status GetOrCreateMLValue(int index, const MLValueAllocationParameters& parameters, T*& value) {
    if (index < 0 || index >= node_values_.size()) {
      return Status(LOTUS, INVALID_ARGUMENT,
                    "Try to access with invalid node value index: " + std::to_string(index));
    }

    if (node_values_[index] < 0) {
      value = nullptr;
      return Status::OK();
    }

    auto p_mlvalue = &all_values_.at(node_values_[index]);

    if (p_mlvalue->IsAllocated()) {
      // The ml has already been allocated.
      // Now only tensor need to be check.
      value = p_mlvalue->GetMutable<T>();
      VerifyShape<T>(value, p_mlvalue, parameters);  // TODO find a better way to do this
      return Status::OK();
    } else {
      // It's not allocated, then allocate it with given shape and return.
      // Perform allocation based on the allocation plan
      LOTUS_RETURN_IF_ERROR(AllocateAsPerAllocationPlan(node_values_[index], parameters));
      value = p_mlvalue->template GetMutable<T>();
      return Status::OK();
    }
  }

  void TraceAllocate(int mlvalue_idx, size_t size);

  void TraceFree(int mlvalue_idx);

  const SequentialExecutionPlan::AllocPlanPerValue& GetAllocationPlan(int mlvalue_idx);

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

  std::unordered_map<string, int> value_name_to_index_;

  const Lotus::SessionState& session_state_;

  // If we already have cached memory pattern on these input shapes
  // Use this mem pattern that create a big chunk for all the internal
  // kernel's input/output tensors.
  const MemoryPatternGroup* mem_patterns_;

  // If no cached memory pattern, and we enable the memory pattern optimization
  // use this planner_ to trace the memory allocation in current executor.
  std::unique_ptr<MLValuePatternPlanner> planner_;

  // Record the ml value indices for output values. we won't include those
  // values' allocation in memory pattern, as they can't be shared.
  std::vector<int> output_indices_;

  // Big chunks on different locations that will be used by mem_pattern.
  std::map<AllocatorInfo, BufferUniquePtr> buffers_;
};
}  // namespace Lotus
