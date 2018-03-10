#ifndef CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
#define CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H

#include <string>
#include <memory>

#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/common/common.h"

namespace Lotus
{
  // Execution provider class name is registered as a provider type.
  typedef std::string ProviderType;
  
  // Map from operator name to kernels.
  typedef OpKernel* (*KernelCreateFn)(OpKernelInfo*);
  
  struct KernelDef {
    // The operator name.
    std::string op_name;
    
    // The type of the execution provider.
    ProviderType provider_type;

    // The data types that each type parameter supports.
    std::unordered_map<std::string, std::vector<MLDataType>> type_constraints;

    // An element <i, j> means that output j reuses the memory of input i.
    std::vector<std::pair<int, int>> inplace_map;

    // An element <i, j> means that output j is an alias of input i.
    std::vector<std::pair<int, int>> alias_map;

    // The inputs/outputs of this kernel that are in host memory.
    std::vector<std::pair<int, bool>> host_memory_args;
  };

  struct KernelCreateInfo {
    std::unique_ptr<KernelDef> kernel_def;
    KernelCreateFn kernel_create_fn;
  };
    
  typedef std::unordered_multimap<std::string, KernelCreateInfo> KernelRegistry;

  KernelRegistry& GlobalKernelRegistry() {
    static KernelRegistry kernel_registry;
    return kernel_registry;
  }

  // TODO KernelRegistry is a multimap and hence to find the right match we need
  // more than just the operator name.
  const KernelCreateInfo* GetOpKernelCreateInfoFromRegistry(const std::string& opId) {
    KernelRegistry& kr = GlobalKernelRegistry();
    auto it = kr.find(opId);
    if (it == kr.end()) {
      return nullptr;
    }

    const KernelCreateInfo& kernel_create_info = it->second;
    return &kernel_create_info;
  }

  class KernelDefBuilder {
  public:
    // Starts with just the name field set.
    explicit KernelDefBuilder(const std::string& op_name)
      : kernel_def_(new KernelDef) {
        kernel_def_->op_name = op_name;
    }

    // The execution provider type of the kernel.
    KernelDefBuilder& Provider(ProviderType provider_type) {
      kernel_def_->provider_type = provider_type;
      return *this;
    }

    // Specify the set of types that this kernel supports. A further restriction      
    // of the set of types specified in the op schema.
    KernelDefBuilder& TypeConstraint(const std::string& attr_name,
                                     std::vector<MLDataType> dtypes) {
      auto& dtypesx = kernel_def_->type_constraints[attr_name];
      for (MLDataType dtype : dtypesx) {
        dtypes.push_back(dtype);
      }
      return *this;
    }

    // Like TypeConstraint but supports just a single type.
    KernelDefBuilder& TypeConstraint(const std::string& attr_name,
                                     MLDataType dtype) {
      auto& dtypes = kernel_def_->type_constraints[attr_name];
      dtypes.push_back(dtype);
      return *this;
    }

    // Like TypeConstraint for type T.
    template <class T>
      KernelDefBuilder& TypeConstraint(const std::string& attr_name) {
      return TypeConstraint(attr_name, DataTypeImpl::GetTensorType<T>());
    }

    // Inplace mapping from inputs to outputs.
    KernelDefBuilder& Inplace(const std::vector<std::pair<int, int>>& inplaces) {
      for (auto& x : inplaces) {
        kernel_def_->inplace_map.push_back(x);
      }
      return *this;
    }

    KernelDefBuilder& Inplace(int i, int j) {
      kernel_def_->inplace_map.push_back({i, j});
      return *this;
    }

    // Alias mapping from inputs to outputs. Different from Inplace that the 
    // content of the tensor is not changed. This is to take care of operators
    // such as Identity and Reshape.
    KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
      for (auto& x : aliases) {
        kernel_def_->alias_map.push_back(x);
      }
      return *this;
    }

    KernelDefBuilder& Alias(int i, int j) {
      kernel_def_->alias_map.push_back({i, j});
      return *this;
    }
  
    // Specify that this kernel requires/provides an input/output arg
    // in host memory (instead of the default, device memory).
    KernelDefBuilder& HostMemory(int index, bool is_input) {
      kernel_def_->host_memory_args.push_back(std::pair<int, bool>(index, is_input));
      return *this;
    }

    // Return the kernel definition.
    std::unique_ptr<KernelDef> Build() {
      return std::move(kernel_def_);
    }
  
  private:
    std::unique_ptr<KernelDef> kernel_def_; // owned only until Build() is invoked.
    LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(KernelDefBuilder);    
  };
}

#endif  // CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
