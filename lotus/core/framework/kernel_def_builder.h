#pragma once

#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/framework/data_types.h"

namespace Lotus {
class KernelDefBuilder;

typedef std::map<int, MemType> MemTypeMap;

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

  LotusIR::ProviderType Provider() const {
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

  const MemTypeMap& MemoryType() const {
    return memory_type_args_;
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
  std::string provider_type_;

  // The supported data types for inputs/outputs.
  // Key is input/output name defined in op schema, Value are supported types.
  std::unordered_map<std::string, std::vector<MLDataType>> type_constraints_;

  // An element <i, j> means that output j reuses the memory of input i.
  std::vector<std::pair<int, int>> inplace_map_;

  // An element <i, j> means that output j is an alias of input i.
  std::vector<std::pair<int, int>> alias_map_;

  // The memory types of inputs/outputs of this kernel
  MemTypeMap memory_type_args_;
};

class KernelDefBuilder {
 public:
  KernelDefBuilder() = default;

  // Starts with just the name field set.
  explicit KernelDefBuilder(const std::string& op_name)
      : kernel_def_(new KernelDef()) {
    kernel_def_->op_name_ = op_name;
  }

  KernelDefBuilder& Domain(const std::string& domain) {
    kernel_def_->op_domain_ = domain;
    return *this;
  }

  // This kernel supports operator definition since <since_version> (to latest).
  KernelDefBuilder& SinceVersion(int since_version) {
    kernel_def_->op_since_version_start_ = since_version;
    return *this;
  }

  // The start and end version should be set accordingly per version range for
  // each domain registered in OpSchemaRegistry::DomainToVersionRange in
  // \Lotus\lotus\core\graph\op.h as below.
  // Key: domain. Value: <lowest version, highest version> pair.
  // std::unordered_map<std::string, std::pair<int, int>> map_;
  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    kernel_def_->op_since_version_start_ = since_version_start;
    kernel_def_->op_since_version_end_ = since_version_end;
    return *this;
  }

  // The execution provider type of the kernel.
  KernelDefBuilder& Provider(LotusIR::ProviderType provider_type) {
    kernel_def_->provider_type_ = provider_type;
    return *this;
  }

  // Specify the set of types that this kernel supports. A further restriction
  // of the set of types specified in the op schema.
  // The arg name could be either op formal parameter name, say "X", or type
  // argument name specified in op schema, say "T".
  KernelDefBuilder& TypeConstraint(const std::string& arg_name,
                                   const std::vector<MLDataType>& supported_types) {
    kernel_def_->type_constraints_[arg_name] = supported_types;
    return *this;
  }

  // Like TypeConstraint but supports just a single type.
  KernelDefBuilder& TypeConstraint(const std::string& arg_name,
                                   MLDataType supported_type) {
    kernel_def_->type_constraints_[arg_name] = std::vector<MLDataType>{supported_type};
    return *this;
  }

  // Inplace mapping from inputs to outputs allowed.
  // It means that uplayer runtime could do memory in-place optimization
  // as it will not impact the correctness of this kernel.
  KernelDefBuilder& MayInplace(const std::vector<std::pair<int, int>>& inplaces) {
    kernel_def_->inplace_map_ = inplaces;
    return *this;
  }

  // allowing output j to reuse memory of input i
  KernelDefBuilder& MayInplace(int i, int j) {
    // TODO: validate inputs.
    kernel_def_->inplace_map_.push_back({i, j});
    return *this;
  }

  // Alias mapping from inputs to outputs. Different from Inplace that the
  // content of the tensor is not changed. This is to take care of operators
  // such as Identity and Reshape.
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    kernel_def_->alias_map_ = aliases;
    return *this;
  }

  KernelDefBuilder& Alias(int i, int j) {
    kernel_def_->alias_map_.push_back({i, j});
    return *this;
  }

  // Specify that this kernel provides an output arg
  // in certain memory type (instead of the default, device memory).
  template <MemType T>
  KernelDefBuilder& MemoryType(int index) {
    kernel_def_->memory_type_args_.insert(std::make_pair(index, T));
    return *this;
  }

  // Return the kernel definition, passing ownership of the KernelDef to the caller
  unique_ptr<KernelDef> Build() {
    int domain_version_start = domain_version_map_.at(kernel_def_->op_domain_).first;
    int domain_version_end = domain_version_map_.at(kernel_def_->op_domain_).second;
    if (kernel_def_->op_since_version_end_ == INT_MAX) {
      kernel_def_->op_since_version_end_ = domain_version_end;
    }
    LOTUS_ENFORCE(kernel_def_->op_since_version_start_ >= domain_version_start && kernel_def_->op_since_version_end_ <= domain_version_end);

    return std::move(kernel_def_);
  }

 private:
  // we own the KernelDef until Build() is called.
  std::unique_ptr<KernelDef> kernel_def_;
  const std::unordered_map<string, std::pair<int, int>>& domain_version_map_ =
      OpSchemaRegistry::DomainToVersionRange::Instance().Map();
};

}  // namespace Lotus
