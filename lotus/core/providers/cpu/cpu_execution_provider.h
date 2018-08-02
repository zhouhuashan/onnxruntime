#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph_transformer.h"
#include "core/graph/constants.h"

namespace Lotus {
// Information needed to construct CPU execution providers.
struct CPUExecutionProviderInfo {
  std::string name;
  bool create_arena;

  explicit CPUExecutionProviderInfo(const char* provider_name, bool use_arena = true)
      : name(provider_name), create_arena(use_arena) {}
  CPUExecutionProviderInfo()
      : CPUExecutionProviderInfo("") {}
};

class CPUTransformer : public LotusIR::GraphTransformer {
 public:
  explicit CPUTransformer(const std::string& name)
      : LotusIR::GraphTransformer(name, "Transformer for CPU execution provider") {
  }

  Status Apply(LotusIR::Graph* /*graph*/, bool* /*modified*/) const override {
    //TODO: any fusing needed on cpu
    return Common::Status::OK();
  }
};

struct KernelCreateInfo;
void RegisterCPUKernels(std::function<void(KernelCreateInfo&&)> create_fn);

// Logical device representation.
class CPUExecutionProvider : public IExecutionProvider {
 public:
  explicit CPUExecutionProvider(const CPUExecutionProviderInfo& info)
      : cpu_transformer_(info.name) {
    DeviceAllocatorRegistrationInfo device_info({kMemTypeDefault, [](int) { return std::make_unique<CPUAllocator>(); }, std::numeric_limits<size_t>::max()});
#ifdef USE_JEMALLOC
    //JEMalloc already has memory pool, so just use device allocator.
    InsertAllocator(kMemTypeDefault,
                    std::shared_ptr<IArenaAllocator>(
                        std::make_unique<DummyArena>(std::move(device_info.factory(0)))));
#else
    if (info.create_arena)
      InsertAllocator(kMemTypeDefault, CreateAllocator(device_info));
    else
      InsertAllocator(kMemTypeDefault,
                      std::shared_ptr<IArenaAllocator>(
                          std::make_unique<DummyArena>(device_info.factory(0))));
#endif
  }

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return cpu_transformer_;
  }

  Status Compute(const LotusIR::Node& node, OpKernelContext* context) const override {
    UNUSED_PARAMETER(node);
    UNUSED_PARAMETER(context);
    return Common::Status(
        Common::LOTUS, Common::FAIL,
        "CPU execution provider: cannot run an op of type `" + node.OpType() + "'.");
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

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const;

 private:
  CPUTransformer cpu_transformer_;
};
}  // namespace Lotus
