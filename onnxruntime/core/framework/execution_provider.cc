// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph.h"
#include "core/framework/computation_capacity.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

const AllocatorMap& IExecutionProvider::GetAllocatorMap() const {
  return allocators_;
}

AllocatorPtr
IExecutionProvider::GetAllocator(MemType mem_type) const {
  if (allocators_.count(mem_type) > 0)
    return allocators_.at(mem_type);
  else
    return nullptr;
}

std::vector<std::unique_ptr<ComputationCapacity>>
IExecutionProvider::GetCapability(const onnxruntime::Graph& graph,
                                  const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputationCapacity>> result;
  for (auto& node : graph.Nodes()) {
    if (graph.IsSourceNode(node) || graph.IsSinkNode(node)) {
      continue;
    }

    for (auto registry : kernel_registries) {
      if (registry->CanExecutionProviderCreateKernel(node, Type())) {
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node.Index());
        result.push_back(std::make_unique<ComputationCapacity>(std::move(sub_graph), nullptr));
      }
    }
  }

  return result;
}

common::Status IExecutionProvider::CopyTensor(const Tensor& src,
                                              Tensor& dst,
                                              int exec_queue_id) const {
  // execution provider may override this to support different exec queues
  ONNXRUNTIME_ENFORCE(exec_queue_id == 0);
  return CopyTensor(src, dst);
}

common::Status IExecutionProvider::Sync() const { return Status::OK(); };

common::Status IExecutionProvider::OnRunStart() { return Status::OK(); }

common::Status IExecutionProvider::OnRunEnd() { return Status::OK(); }

void IExecutionProvider::InsertAllocator(MemType mem_type,
                                         AllocatorPtr allocator) {
  allocators_.insert({mem_type, allocator});
}
}  // namespace onnxruntime
