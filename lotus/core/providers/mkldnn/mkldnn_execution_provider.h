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

class MKLDNNTransformer : public LotusIR::GraphTransformer {
 public:
  MKLDNNTransformer(const std::string& name);
  Status Apply(LotusIR::Graph* graph, bool* modified) const override;
};

// Logical device representation.
class MKLDNNExecutionProvider : public IExecutionProvider {
 public:
  explicit MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info);
  virtual ~MKLDNNExecutionProvider();

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return transformer_;
  }

  Status Compute(const LotusIR::Node& node, OpKernelContext* /*context*/) const override {
    return Common::Status(
        Common::LOTUS, Common::FAIL,
        "MKLDNN execution provider: can not run an op of type `" + node.OpType() + "'.");
  }

  std::string Type() const override {
    return LotusIR::kMklDnnExecutionProvider;
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const;

 private:
  MKLDNNTransformer transformer_;
};

}  // namespace Lotus
