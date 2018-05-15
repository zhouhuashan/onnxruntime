#pragma once

#include <unordered_map>
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
  virtual ~IExecutionProvider() {}

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
  virtual Status CopyTensor(const Tensor& src, Tensor& dst) const = 0;

  // Returns an opaque handle whose exact type varies based on the provider
  // and is interpreted accordingly by the corresponding kernel implementation.
  // For Direct3D operator kernels, this may return an IUnknown supporting
  // QueryInterface to ID3D12GraphicsCommandList1.
  virtual const void* GetExecutionHandle() const noexcept = 0;

  // @return type of the execution provider; should match that set in the node through
  // the SetExecutionProvider API.
  // Example valid return values are: kCpuExecutionProvider, kCudaExecutionProvider
  virtual std::string Type() const = 0;

 protected:
  AllocatorMap allocators_;
};
}  // namespace Lotus
