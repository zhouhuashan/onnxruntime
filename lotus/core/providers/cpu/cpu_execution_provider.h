#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"

namespace Lotus {
class DummyCPUTransformer : public LotusIR::IGraphTransformer {
 public:
  virtual Status Apply(/*IN/OUT*/ LotusIR::Graph& graph, /*OUT*/ bool& modified) override {
    auto num_nodes = graph.NumberOfNodes();
    for (int i = 0; i < num_nodes; i++) {
      if (graph.IsSourceNode(i) || graph.IsSinkNode(i))
        continue;
      auto node = graph.GetNode(i);
      if (node->GetExecutionProvider().empty()) {
        node->SetExecutionProvider(LotusIR::kCpuExecutionProvider);
        modified = true;
      }
    }

    return Common::Status::OK();
  }
};

// Logical device representation.
class CPUExecutionProvider : public IExecutionProvider {
 public:
  explicit CPUExecutionProvider(const ExecutionProviderInfo& info) {
    UNUSED_PARAMETER(info);
  }

  virtual LotusIR::IGraphTransformer& GetTransformer() override {
    return dummy_transformer_;
  }

  virtual IArenaAllocator& GetTempSpaceAllocator() const override {
    auto& alloc_mgr = AllocatorManager::Instance();
    return alloc_mgr.GetArena(CPU);
  }

  virtual Common::Status Compute(const LotusIR::Node& node, OpKernelContext* context) override {
    UNUSED_PARAMETER(node);
    UNUSED_PARAMETER(context);
    //LOTUS_NOT_IMPLEMENTED;
    return Common::Status::OK();
  }

  virtual Status CopyTensor(const Tensor& src, Tensor& dst) override {
    LOTUS_ENFORCE(dst.Location().name == CPU);

    // Todo: support copy with different devices.
    if (src.Location().name != CPU)
      LOTUS_NOT_IMPLEMENTED;

    // no really copy needed if is copy to cpu.
    dst.ShallowCopy(src);

    return Status::OK();
  }

 private:
  DummyCPUTransformer dummy_transformer_;
};
}  // namespace Lotus
