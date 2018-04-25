#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"

namespace Lotus {

class CPUTransformer : public LotusIR::GraphTransformer {
 public:
  CPUTransformer(const std::string& name)
      : LotusIR::GraphTransformer(name, "Transformer for CPU execution provider") {
  }

  Status Apply(LotusIR::Graph* graph, bool* modified) const override {
    for (auto& node : graph->Nodes()) {
      if (graph->IsSourceNode(node) || graph->IsSinkNode(node))
        continue;

      if (node.GetExecutionProvider().empty()) {
        node.SetExecutionProvider(LotusIR::kCpuExecutionProvider);
        *modified = true;
      }
    }

    return Common::Status::OK();
  }
};

// Logical device representation.
class CPUExecutionProvider : public IExecutionProvider {
 public:
  explicit CPUExecutionProvider(const ExecutionProviderInfo& info)
      : cpu_transformer_(info.name) {
  }

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return cpu_transformer_;
  }

  IArenaAllocator& GetTempSpaceAllocator() override {
    auto& alloc_mgr = AllocatorManager::Instance();
    return alloc_mgr.GetArena(CPU);
  }

  Status Compute(const LotusIR::Node& node, OpKernelContext* context) const override {
    UNUSED_PARAMETER(node);
    UNUSED_PARAMETER(context);
    return Common::Status(
        LOTUS, FAIL,
        "CPU execution provider: can not run an op of type `" + node.OpType() + "'.");
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) override {
    LOTUS_ENFORCE(dst.Location().name == CPU);

    // Todo: support copy with different devices.
    if (src.Location().name != CPU)
      LOTUS_NOT_IMPLEMENTED("copy to ", src.Location().name, " is not implemented");

    // no really copy needed if is copy to cpu.
    dst.ShallowCopy(src);

    return Status::OK();
  }

 private:
  CPUTransformer cpu_transformer_;
};
}  // namespace Lotus
