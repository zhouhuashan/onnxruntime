#pragma once

#include <unordered_map>
#include "core/framework/arena.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"

namespace Lotus {

class OpKernelContext;

// Information needed to construct execution providers.  
struct ExecutionProviderInfo {
  std::string name;
};

// Logical device representation.
class IExecutionProvider {
 public:
  virtual ~IExecutionProvider() {}

  // Graph to graph transformation. The resulting graph may contain custom
  // operators introduced by this execution provider. Newly formed custom
  // functions must be registered in kernelRegistry_.
  virtual const LotusIR::GraphTransformer& GetTransformer() const = 0;

  // Get IAllocator for <*this> execution provider.
  // It will be used for allocating tensors (inputs/outputs) or copying tensors
  // (IN/OUT) for this execution provider.
  virtual IArenaAllocator& GetTempSpaceAllocator() = 0;

  // Run the computation of a given node.
  virtual Common::Status Compute(const LotusIR::Node& node, OpKernelContext* context) const = 0;

  // TODO: Do we still need these copy methods?
  // TODO: Shouldn't tensor copy be implemented in the Tensor class, with it optionally taking
  // a parameter to provide device specific logic?
  virtual Status CopyTensor(const Tensor& src, Tensor& dst) = 0;

  // Returns an opaque handle whose exact type varies based on the provider
  // and is interpreted accordingly by the corresponding kernel implementation.
  // For Direct3D operator kernels, this may return an IUnknown supporting
  // QueryInterface to ID3D12GraphicsCommandList1.
  virtual const void* GetExecutionHandle() const noexcept = 0;
};

typedef std::function<unique_ptr<IExecutionProvider>(const ExecutionProviderInfo&)> ProviderCreateFunc;
typedef std::unique_ptr<IExecutionProvider> ExecutionProviderPtr;

// Singleton execution provider manager.
// It holds a global provider type to provider finder map, and will find/create
// execution provider instances for inference engine.
class ExecutionProviderMgr {
 public:
  static ExecutionProviderMgr& Instance() {
    static ExecutionProviderMgr instance;
    return instance;
  }

  // TODO: registration for provider type to provider finder.
  Status AddProviderCreater(const string& key, ProviderCreateFunc creator_func) {
    if (provider_map_.find(key) == provider_map_.end()) {
      provider_map_[key] = creator_func;
      return Status::OK();
    } else {
      LOTUS_THROW("Execution provider () " + key + " already registered");
      // return Status(LOTUS, INVALID_ARGUMENT, "Execution provider () " + key + " already registered");
    }
  }

  ExecutionProviderPtr GetProvider(const string& key, const ExecutionProviderInfo& info) {
    auto iter = provider_map_.find(key);
    if (iter == provider_map_.end()) {
      return nullptr;
    }

    return iter->second(info);
  }

 private:
  ExecutionProviderMgr() {}

  std::unordered_map<std::string, ProviderCreateFunc> provider_map_;
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
