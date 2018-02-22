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
  
  // Logical device represenatation.
  class IExecutionProvider
  {
  public:
    IExecutionProvider()
    {
      m_id = Name() + "." + Version();
    }

    virtual ~IExecutionProvider() {}

    virtual const std::string& Name() const = 0;
    
    virtual const std::string& Version() const = 0;

    virtual const std::string& ID() const
    {
      return m_id;
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

  private:
    std::string m_id;
  };

  class ExecutionProviderInfo {
  };
  
  typedef IExecutionProvider* (*ProviderCreateFn)(ExecutionProviderInfo*);

  // Singleton execution provider manager.
  // It holds a global provider type to provider finder map, and will find/create
  // execution provider instances for inference engine.
  class ExecutionProviderMgr
  {
  public:
    static ExecutionProviderMgr Instance()
    {
      static ExecutionProviderMgr s_providerMgr;
      return s_providerMgr;
    }

    // TODO: registration for provider type to provider finder.

  private:
    ExecutionProviderMgr() {}

    std::unordered_map<std::string, ProviderCreateFn> provider_map_;
  };
}
#endif  // CORE_FRAMEWORK_EXECUTION_PROVIDER_H
