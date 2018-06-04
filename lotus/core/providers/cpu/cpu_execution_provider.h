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

      if (node.GetExecutionProviderType().empty()) {
        node.SetExecutionProviderType(LotusIR::kCpuExecutionProvider);
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
    if (cpu_allocator_creator != device_factories.end()) {
#ifdef USE_JEMALLOC
      //JEMalloc already has memory pool, so just use device allocator.
      allocators_.insert(std::make_pair(kMemTypeDefault,
                                        std::shared_ptr<IArenaAllocator>(
                                            std::make_unique<DummyArena>(std::move(cpu_allocator_creator->second.factory(0))))));
#else
      allocators_.insert(std::make_pair(kMemTypeDefault, CreateAllocator(cpu_allocator_creator->second)));
#endif
    }
  }

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return cpu_transformer_;
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

  Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    LOTUS_ENFORCE(dst.Location().name == CPU);

    // Todo: support copy with different devices.
    if (src.Location().name != CPU)
      LOTUS_NOT_IMPLEMENTED("copy to ", src.Location().name, " is not implemented");

    // no really copy needed if is copy to cpu.
    dst.ShallowCopy(src);

    return Status::OK();
  }

  const void* GetExecutionHandle() const noexcept override {
    // The CPU interface does not return anything interesting.
    return nullptr;
  }

 private:
  CPUTransformer cpu_transformer_;
};
}  // namespace Lotus
