#ifndef CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
#define CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H

#include <string>
#include "core/framework/data_types.h"

// The types of execution providers.
enum ProviderType {
    kCPU = 1,
    kDirectML = 2,
    kCUDA = 3,
    kFPGA = 4;
    kGraphCore = 5;
    kNNAPI = 6;
    kCoreML = 7;
};

struct KernelDef {
public:
  std::string op_name;
  ProviderType provider_type;
  std::unordered_map<std::string, std::vector<MLDataType>> type_constraints;
  std::vector<std::string> host_memory_args;
  std::vector<std::pair<int, int>> inplace_map;
  std::vector<std::pair<int, int>> alias_map;
}

class KernelDefBuilder {
public:
  // Starts with just the name field set.
  explicit KernelDefBuilder(const std::string& op_name) {
    m_kernel_def = new KernelDef;
    m_kernel_def->op_name = op_name;
  }

  // The execution provider type of the kernel.
  KernelDefBuilder& Provider(ProviderType provider_type) {
    m_kernel_def->provider_type = provider_type;
    return *this;
  }

  // Specify the set of types that this kernel supports. A further restriction      
  // of the set of types specified in the op schema.
  KernelDefBuilder& TypeConstraint(const std::string& attr_name,
                                   std::vector<MLDataType> dtypes) {
    auto& dtypes = m_kernel_def->type_constraints[attr_name];
    for (MLDataType dtype : dtypes) {
      dtypes.push_back(dtype);
    }
    return *this;
  }

  // Like TypeConstraint but supports just a single type.
  KernelDefBuilder& TypeConstraint(const std::string& attr_name,
                                   MLDataType dtype) {
    auto& dtypes = m_kernel_def->type_constraints[attr_name];
    dtypes.push_back(dtype);
    return *this;
  }

  // Like TypeConstraint for type T.
  template <class T>
  KernelDefBuilder& TypeConstraint(const std::string& attr_name) {
    return TypeConstraint(attr_name, DataTypeImpl::GetType<T>());
  }

  // Inplace mapping from inputs to outputs.
  KernelDefBuilder& Inplace(const std::vector<std::pair<int, int>>& inplaces) {
    for (auto& x : inplaces) {
      m_kernel_def->inplace_map.push_back(x);
    }
    return *this;
  }

  // Alias mapping from inputs to outputs. Different from Inplace that the 
  // content of the tensor is not changed. This is to take care of operators
  // such as Identity and Reshape.
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    for (auto& x : aliases) {
      m_kernel_def->alias_map.push_back(x);
    }
    return *this;
  }

  // Specify that this kernel requires/provides an input/output arg
  // in host memory (instead of the default, device memory).
  KernelDefBuilder& HostMemory(const std::string& arg_name) {
    m_kernel_def->host_memory_args.push_back(arg_name);
    return *this;
  }

private:
  KernelDef* kernelDef_;   // not owned.
};

#endif  // CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
