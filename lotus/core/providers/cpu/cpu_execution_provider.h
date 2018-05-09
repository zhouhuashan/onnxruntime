#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"

namespace Lotus {
// Information needed to construct CPU execution providers.
struct CPUExecutionProviderInfo {
  std::string name;
};

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
  explicit CPUExecutionProvider(const CPUExecutionProviderInfo& info)
      : cpu_transformer_(info.name) {
    auto& device_factories = DeviceAllocatorRegistry::Instance().AllRegistrations();
    auto cpu_allocator_creator = device_factories.find(CPU);
    if (cpu_allocator_creator != device_factories.end())
      arena_ = std::move(CreateArena(cpu_allocator_creator->second ));
  }

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return cpu_transformer_;
  }

  AllocatorPtr GetAllocator() override {
    return arena_;
  }

  Status Compute(const LotusIR::Node& node, OpKernelContext* context) const override {
    UNUSED_PARAMETER(node);
    UNUSED_PARAMETER(context);
    return Common::Status(
        LOTUS, FAIL,
        "CPU execution provider: can not run an op of type `" + node.OpType() + "'.");
  }

  std::string Type() const override {
    return LotusIR::kCpuExecutionProvider;
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

  virtual const void* GetExecutionHandle() const noexcept override {
    // The CPU interface does not return anything interesting.
    return nullptr;
  }

 private:
  CPUTransformer cpu_transformer_;
  ArenaPtr arena_;
};
}  // namespace Lotus
