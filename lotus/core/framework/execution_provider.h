#ifndef CORE_FRAMEWORK_EXECUTION_PROVIDER_H
#define CORE_FRAMEWORK_EXECUTION_PROVIDER_H

#include <unordered_map>
#include "core/framework/arena.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"

using namespace LotusIR;

namespace Lotus {
typedef std::shared_ptr<void> ExecutionProviderInfo;

// Logical device represenatation.
class IExecutionProvider {
 public:
  virtual ~IExecutionProvider() {}

  // Graph to graph transformation. The resulting graph may contain custom
  // operators introduced by this execution provider. Newly formed custom
  // functions must be registered in kernelRegistry_.
  virtual IGraphTransformer& GetTransformer() = 0;

  // Get IAllocator for <*this> execution provider.
  // It will be used for allocating tensors (inputs/outputs) or copying tensors
  // (IN/OUT) for this execution provider.
  virtual IArenaAllocator& GetTempSpaceAllocator() const = 0;

  // Run the computation of a given node.
  virtual Common::Status Compute(const Node& node, OpKernelContext* context) = 0;

  // TODO: Do we still need these copy methods?
  virtual Status CopyTensorTo(const Tensor& srcTensor,
                              Tensor* p_dstTensor) = 0;

  virtual Status CopyTensorFrom(const Tensor& srcTensor,
                                Tensor* p_dstTensor) = 0;
};

typedef std::function<unique_ptr<IExecutionProvider>(const ExecutionProviderInfo&)> ProviderCreateFn;
typedef std::unique_ptr<IExecutionProvider> ExecutionProviderPtr;

// Singleton execution provider manager.
// It holds a global provider type to provider finder map, and will find/create
// execution provider instances for inference engine.
class ExecutionProviderMgr {
 public:
  static ExecutionProviderMgr& Instance() {
    static ExecutionProviderMgr s_providerMgr;
    return s_providerMgr;
  }

  // TODO: registration for provider type to provider finder.
  Status AddProviderCreater(const string& key, ProviderCreateFn creatorFn) {
    if (provider_map_.find(key) == provider_map_.end()) {
      provider_map_[key] = creatorFn;
      return Status::OK();
    } else {
      LOTUS_ENFORCE(false, "Execution provider () " + key + " already registered");
      return Status(LOTUS, INVALID_ARGUMENT, "Execution provider () " + key + " already registered");
    }
  }

  ExecutionProviderPtr GetProvider(const string& key, const ExecutionProviderInfo& info) {
    if (provider_map_.find(key) == provider_map_.end()) {
      return nullptr;
    } else {
      return ExecutionProviderPtr(provider_map_[key](info));
    }
  }

 private:
  ExecutionProviderMgr() {}

  std::unordered_map<std::string, ProviderCreateFn> provider_map_;
};

// Execution provider registration macro.
// It registers a provider with provider class name.
#define REGISTER_PROVIDER(...) \
  REGISTER_PROVIDER_HELPER(__COUNTER__, __VA_ARGS__)
#define REGISTER_PROVIDER_HELPER(Counter, ...)                                                                             \
  namespace {                                                                                                              \
  static Status s_##Counter = ExecutionProviderMgr::Instance()                                                             \
                                  .AddProviderCreater(#__VA_ARGS__,                                                        \
                                                      [](const ExecutionProviderInfo& info)                                \
                                                          -> unique_ptr<IExecutionProvider> {                              \
                                                        return std::unique_ptr<IExecutionProvider>(new __VA_ARGS__(info)); \
                                                      });                                                                  \
  }
}  // namespace Lotus
#endif  // CORE_FRAMEWORK_EXECUTION_PROVIDER_H
