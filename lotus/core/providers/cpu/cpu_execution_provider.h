#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph_transformer.h"
#include "core/graph/constants.h"
#include "core/providers/provider_factories.h"

namespace onnxruntime {

using FuseRuleFn = std::function<void(const onnxruntime::Graph&, std::vector<std::unique_ptr<ComputationCapacity>>&)>;
// Logical device representation.
class CPUExecutionProvider : public IExecutionProvider {
 public:
  explicit CPUExecutionProvider(const CPUExecutionProviderInfo& info) {
    UNUSED_PARAMETER(info);
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

  std::string Type() const override {
    return onnxruntime::kCpuExecutionProvider;
  }

  virtual std::vector<std::unique_ptr<ComputationCapacity>>
  GetCapability(const onnxruntime::Graph& graph,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

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

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  void InsertFusedRules(FuseRuleFn rule);

 protected:
  std::vector<FuseRuleFn> fuse_rules_;
};
}  // namespace onnxruntime
