#pragma once
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph_transformer.h"

namespace Lotus {

struct MKLDNNExecutionProviderInfo {
  std::string name;
  bool create_arena;

  MKLDNNExecutionProviderInfo(const char* provider_name, bool use_arena = true)
      : name(provider_name), create_arena(use_arena) {}
  MKLDNNExecutionProviderInfo()
      : MKLDNNExecutionProviderInfo("") {}
};

// Logical device representation.
class MKLDNNExecutionProvider : public IExecutionProvider {
 public:
  explicit MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info);
  virtual ~MKLDNNExecutionProvider();

  std::string Type() const override {
    return LotusIR::kMklDnnExecutionProvider;
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace Lotus
