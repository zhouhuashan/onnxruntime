#ifndef CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
#define CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H

#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/framework/data_types.h"

namespace Lotus {
// Execution provider class name is registered as a provider type.
typedef std::string ProviderType;

class KernelDefBuilder;

class KernelDef {
 public:
  const std::string& OpName() const {
    return op_name_;
  }

  const std::string& Domain() const {
    return op_domain_;
  }

  void SinceVersion(/*out*/ int* start, /*out*/ int* end) const {
    *start = op_since_version_start_;
    *end = op_since_version_end_;
  }

  const ProviderType& Provider() const {
    return provider_type_;
  }

  const std::unordered_map<std::string, std::vector<MLDataType>>& TypeConstraints() const {
    return type_constraints_;
  }

  const std::vector<std::pair<int, int>>& MayInplace() const {
    return inplace_map_;
  }

  const std::vector<std::pair<int, int>>& Alias() const {
    return alias_map_;
  }

  const std::vector<std::pair<int, bool>>& HostMemory() const {
    return host_memory_args_;
  }

 private:
  friend class KernelDefBuilder;
  
  // The operator name supported by <*this> kernel..
  std::string op_name_;

  // The operator since_version range supported by <*this> kernel.
  // A kernel could support an operator definition between <op_since_version_start>
  // and <op_since_version_end> (inclusive).
  int op_since_version_start_ = 1;
  int op_since_version_end_ = INT_MAX;

  // THe operator domain supported by <*this> kernel.
  std::string op_domain_ = LotusIR::kOnnxDomain;

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

class KernelDefBuilder {
public:
  KernelDefBuilder() = default;
  
  // Starts with just the name field set.
  explicit KernelDefBuilder(const std::string& op_name)
    : kernelDef_(new KernelDef()) {
      kernelDef_->op_name_ = op_name;
  }

  KernelDefBuilder& Domain(const std::string& domain) {
    kernelDef_->op_domain_ = domain;
    return *this;
  }

  // This kernel supports operator definition since <since_version> (to latest).
  KernelDefBuilder& SinceVersion(int since_version) {
    kernelDef_->op_since_version_start_ = since_version;
    kernelDef_->op_since_version_end_ = INT_MAX;
    return *this;
  }

  // The start and end version should be set accordingly per version range for
  // each domain registered in OpSchemaRegistry::DomainToVersionRange in
  // \Lotus\lotus\core\graph\op.h as below.
  // Key: domain. Value: <lowest version, highest version> pair.
  // std::unordered_map<std::string, std::pair<int, int>> map_;
  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    kernelDef_->op_since_version_start_ = since_version_start;
    kernelDef_->op_since_version_end_ = since_version_end;
    return *this;
  }

  // The execution provider type of the kernel.
  KernelDefBuilder& Provider(const ProviderType& provider_type) {
    kernelDef_->provider_type_ = provider_type;
    return *this;
  }

  // Specify the set of types that this kernel supports. A further restriction
  // of the set of types specified in the op schema.
  // The arg name could be either op formal parameter name, say "X", or type
  // argument name specified in op schema, say "T".
  KernelDefBuilder& TypeConstraint(const std::string& arg_name,
                            const std::vector<MLDataType>& supported_types) {
    kernelDef_->type_constraints_[arg_name] = supported_types;
    return *this;
  }

  // Like TypeConstraint but supports just a single type.
  KernelDefBuilder& TypeConstraint(const std::string& arg_name,
                                   MLDataType supported_type) {
    kernelDef_->type_constraints_[arg_name] = std::vector<MLDataType>{supported_type};
    return *this;
  }

  // Inplace mapping from inputs to outputs allowed.
  // It means that uplayer runtime could do memory in-place optimization
  // as it will not impact the correctness of this kernel.
  KernelDefBuilder& MayInplace(const std::vector<std::pair<int, int>>& inplaces) {
    kernelDef_->inplace_map_ = inplaces;
    return *this;
  }

  // allowing output j to reuse memory of input i
  KernelDefBuilder& MayInplace(int i, int j) {
    // TODO: validate inputs.
    kernelDef_->inplace_map_.push_back({i, j});
    return *this;
  }

  // Alias mapping from inputs to outputs. Different from Inplace that the
  // content of the tensor is not changed. This is to take care of operators
  // such as Identity and Reshape.
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    kernelDef_->alias_map_ = aliases;
    return *this;
  }

  KernelDefBuilder& Alias(int i, int j) {
    kernelDef_->alias_map_.push_back({i, j});
    return *this;
  }

  // Specify that this kernel requires/provides an input/output arg
  // in host memory (instead of the default, device memory).
  KernelDefBuilder& HostMemory(int index, bool is_input) {
    kernelDef_->host_memory_args_.push_back({index, is_input});
    return *this;
  }

  // Return the kernel definition.
  const KernelDef* Build() {
    return kernelDef_.release();
  }
  
private:
  std::unique_ptr<KernelDef> kernelDef_;   // not owned.
};

}  // namespace Lotus

#endif  // CORE_FRAMEWORK_KERNEL_DEF_BUILDER_H
