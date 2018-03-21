#ifndef CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
#define CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H

#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/framework/data_types.h"

namespace Lotus {
// Execution provider class name is registered as a provider type.
typedef std::string ProviderType;

class KernelDef {
 public:
  KernelDef() = default;

  // Starts with just the name field set.
  explicit KernelDef(const std::string& op_name) {
    op_name_ = op_name;
  }

  const std::string& OpName() const {
    return op_name_;
  }

  KernelDef& Domain(const std::string& domain) {
    op_domain_ = domain;
    return *this;
  }

  const std::string& Domain() const {
    return op_domain_;
  }

  // This kernel supports operator definition since <since_version> (to latest).
  KernelDef& SinceVersion(int since_version) {
    op_since_version_start_ = since_version;
    op_since_version_end_ = INT_MAX;
    return *this;
  }

  // The start and end version should be set accordingly per version range for
  // each domain registered in OpSchemaRegistry::DomainToVersionRange in
  // \Lotus\lotus\core\graph\op.h as below.
  // Key: domain. Value: <lowest version, highest version> pair.
  // std::unordered_map<std::string, std::pair<int, int>> m_map;
  KernelDef& SinceVersion(int since_version_start, int since_version_end) {
    op_since_version_start_ = since_version_start;
    op_since_version_end_ = since_version_end;
    return *this;
  }

  void SinceVersion(/*out*/ int* start, /*out*/ int* end) const {
    *start = op_since_version_start_;
    *end = op_since_version_end_;
  }

  // The execution provider type of the kernel.
  KernelDef& Provider(const ProviderType& provider_type) {
    provider_type_ = provider_type;
    return *this;
  }

  const ProviderType& Provider() const {
    return provider_type_;
  }

  // Specify the set of types that this kernel supports. A further restriction
  // of the set of types specified in the op schema.
  // The arg name could be either op formal parameter name, say "X", or type
  // argument name specified in op schema, say "T".
  KernelDef& TypeConstraint(const std::string& arg_name,
                            const std::vector<MLDataType>& supported_types) {
    type_constraints_[arg_name] = supported_types;
    return *this;
  }

  // Like TypeConstraint but supports just a single type.
  KernelDef& TypeConstraint(const std::string& arg_name,
                            MLDataType supported_type) {
    type_constraints_[arg_name] = std::vector<MLDataType>{supported_type};
    return *this;
  }

  const std::unordered_map<std::string, std::vector<MLDataType>>& TypeConstraints() const {
    return type_constraints_;
  }

  // Inplace mapping from inputs to outputs allowed.
  // It means that uplayer runtime could do memory in-place optimization
  // as it will not impact the correctness of this kernel.
  KernelDef& MayInplace(const std::vector<std::pair<int, int>>& inplaces) {
    inplace_map_ = inplaces;
    return *this;
  }

  KernelDef& MayInplace(int i, int j) {
    // TODO: validate inputs.
    inplace_map_.push_back({i, j});
    return *this;
  }

  const std::vector<std::pair<int, int>>& MayInplace() const {
    return inplace_map_;
  }

  // Alias mapping from inputs to outputs. Different from Inplace that the
  // content of the tensor is not changed. This is to take care of operators
  // such as Identity and Reshape.
  KernelDef& Alias(const std::vector<std::pair<int, int>>& aliases) {
    alias_map_ = aliases;
    return *this;
  }

  KernelDef& Alias(int i, int j) {
    alias_map_.push_back({i, j});
    return *this;
  }

  const std::vector<std::pair<int, int>>& Alias() const {
    return alias_map_;
  }

  // Specify that this kernel requires/provides an input/output arg
  // in host memory (instead of the default, device memory).
  KernelDef& HostMemory(int index, bool is_input) {
    host_memory_args_.push_back({index, is_input});
    return *this;
  }

  const std::vector<std::pair<int, bool>>& HostMemory() const {
    return host_memory_args_;
  }

 private:
  // The operator name supported by <*this> kernel..
  std::string op_name_;

  // The operator since_version range supported by <*this> kernel.
  // A kernel could support an operator definition between <op_since_version_start>
  // and <op_since_version_end> (inclusive).
  int op_since_version_start_ = 1;
  int op_since_version_end_ = INT_MAX;

  // THe operator domain supported by <*this> kernel.
  std::string op_domain_ = LotusIR::c_onnxDomain;

  // The type of the execution provider.
  ProviderType provider_type_;

  // The supported data types for inputs/outputs.
  // Key is input/output name defined in op schema, Value are supported types.
  std::unordered_map<std::string, std::vector<MLDataType>> type_constraints_;

  // An element <i, j> means that output j reuses the memory of input i.
  std::vector<std::pair<int, int>> inplace_map_;

  // An element <i, j> means that output j is an alias of input i.
  std::vector<std::pair<int, int>> alias_map_;

  // The inputs/outputs of this kernel that are in host memory.
  std::vector<std::pair<int, bool>> host_memory_args_;
};

}  // namespace Lotus

#endif  // CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
