#ifndef CORE_FRAMEWORK_EXECUTION_PROVIDER_H
#define CORE_FRAMEWORK_EXECUTION_PROVIDER_H

#include <unordered_map>
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"

using namespace LotusIR;

namespace Lotus
{
  // Map from operator name to kernels. 
  typedef OpKernel* (*KernelCreateFn)(OpKernelInfo*);
  typedef std::unordered_multimap<std::string, KernelCreateFn> KernelRegistry;

  typedef void* EPAdditionalInfo;

  class ExecutionProviderInfo 
  {
  public:
      const string& Name() const { return name_; }
      const string& Version() const { return version_; }
      EPAdditionalInfo AdditionalInfo() { return info_; }

      ExecutionProviderInfo(const string& name, 
          const string& version, 
          EPAdditionalInfo info)
          : name_(name), version_(version), info_(info)
      {}

  private:
      string name_;
      string version_;
      EPAdditionalInfo info_;
  };
  
  // Logical device represenatation.
  class IExecutionProvider
  {
  public:
    IExecutionProvider()
    {
    }

    virtual ~IExecutionProvider() {}

    virtual const std::string& Name() const = 0;
    
    virtual const std::string& Version() const = 0;

    virtual const std::string& ID() const
    {
      return id_;
    }

    // Graph to graph transformation. The resulting graph may contain custom 
    // operators introduced by this execution provider. Newly formed custom
    // functions must be registered in kernelRegistry_. 
    virtual IGraphTransformer& GetTransformer() const = 0;

    // Get IAllocator for <*this> execution provider.
    // It will be used for allocating tensors (inputs/outputs) or copying tensors
    // (IN/OUT) for this execution provider.
    virtual IArenaAllocator& GetAllocator() const = 0;

    // Run the computation of a given node.
    virtual void Compute(Node* node, OpKernelContext* context) = 0;

    // TODO: Do we still need these copy methods?
    virtual Status CopyCPUTensorTo(const Tensor& srcTensor,
                                   Tensor* p_dstTensor) = 0;

    virtual Status CopyTensorToCPU(const Tensor& srcTensor,
                                   Tensor* p_dstTensor) = 0;
    
  protected:
    void SetId()
    {
        id_ = Name() + "." + Version();
    }

    std::string id_;
  };
    
  typedef std::function<unique_ptr<IExecutionProvider>(const ExecutionProviderInfo*)> ProviderCreateFn;

  // Singleton execution provider manager.
  // It holds a global provider type to provider finder map, and will find/create
  // execution provider instances for inference engine.
  class ExecutionProviderMgr
  {
  public:
    static ExecutionProviderMgr& Instance()
    {
      static ExecutionProviderMgr s_providerMgr;
      return s_providerMgr;
    }

    // TODO: registration for provider type to provider finder.
    Status AddProviderCreater(const string& key, ProviderCreateFn creatorFn)
    {
        if (provider_map_.find(key) == provider_map_.end())
        {
            provider_map_[key] = creatorFn;
            return Status::OK();
        }
        else
        {
            return Status(LOTUS, INVALID_ARGUMENT, "Execution provider already registered");
        }
    }

    ProviderCreateFn GetProvider(const string& key)
    {
        if (provider_map_.find(key) == provider_map_.end())
        {
            return nullptr;
        }
        else
        {
            return provider_map_[key];
        }
    }

  private:
    ExecutionProviderMgr() {}

    std::unordered_map<std::string, ProviderCreateFn> provider_map_;
  };
  
#define REGISTRY_PROVIDER_CREATOR(Key, Func) \
  REGISTRY_PROVIDER_CREATOR_HELPER(__COUNTER__, Key, Func)
#define REGISTRY_PROVIDER_CREATOR_HELPER(Counter, Key, Func)          \
  namespace {                                                         \
      static Status s_##Counter = ExecutionProviderMgr::Instance()   \
          .AddProviderCreater(#Key, Func);                             \
  }
  
}
#endif  // CORE_FRAMEWORK_EXECUTION_PROVIDER_H
