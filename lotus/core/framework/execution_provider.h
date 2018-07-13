#pragma once

#include <unordered_map>
#include "core/common/status.h"
#include "core/framework/arena.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"

namespace Lotus {

class OpKernelContext;

// Logical device representation.
typedef std::map<MemType, AllocatorPtr> AllocatorMap;

class IExecutionProvider {
 public:
  virtual ~IExecutionProvider() = default;

  // Graph to graph transformation. The resulting graph may contain custom
  // operators introduced by this execution provider. Newly formed custom
  // functions must be registered in kernelRegistry_.
  virtual const LotusIR::GraphTransformer& GetTransformer() const = 0;

  // Get all IAllocators for <*this> execution provider.
  const AllocatorMap& GetAllocatorMap() const {
    return allocators_;
  }

  // Get allocator with specified MemType
  virtual AllocatorPtr GetAllocator(MemType mem_type = kMemTypeDefault) const {
    return allocators_.at(mem_type);
  }

  // Run the computation of a given node.
  virtual Common::Status Compute(const LotusIR::Node& node, OpKernelContext* context) const = 0;

  // Copy tensor between execution providers
  virtual Common::Status CopyTensor(const Tensor& src, Tensor& dst) const = 0;

  // Returns an opaque handle whose exact type varies based on the provider
  // and is interpreted accordingly by the corresponding kernel implementation.
  // For Direct3D operator kernels, this may return an IUnknown supporting
  // QueryInterface to ID3D12GraphicsCommandList1.
  virtual const void* GetExecutionHandle() const noexcept = 0;

  // @return type of the execution provider; should match that set in the node through
  // the SetExecutionProvider API.
  // Example valid return values are: kCpuExecutionProvider, kCudaExecutionProvider
  virtual std::string Type() const = 0;

  /**
  * Blocks until the device has completed all preceding requested tasks.
  * Currently this is primarily used by the IOBinding object to ensure that all inputs have been
  * copied to the device before execution begins.
  */
  virtual Common::Status Sync() {
    return Status::OK();
  };

  // Called when InferenceSession::Run started
  // NOTE that due to async execution in provider, the actual work of previous Run may not be finished on device
  // This function should be regarded as the point after which a new Run would start to submit commands from CPU
  virtual Common::Status OnRunStart() {
    return Status::OK();
  }

  // Called when InferenceSession::Run ended
  // NOTE that due to async execution in provider, the actual work of this Run may not be finished on device
  // This function should be regarded as the point that all commands of current Run has been submmited by CPU
  virtual Common::Status OnRunEnd() {
    return Status::OK();
  }

 protected:
  AllocatorMap allocators_;
};
}  // namespace Lotus
