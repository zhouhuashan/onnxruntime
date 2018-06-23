#pragma once

#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/framework/data_types.h"

namespace Lotus {
class KernelDefBuilder;

typedef std::map<int, MemType> MemTypeMap;

// note that input/output might be on CPU implicitly when the node is from CPU execution provider
inline bool MemTypeOnCpuExplicitly(const MemTypeMap& mem_type_map, int index) {
  auto iter = mem_type_map.find(index);
  return iter != mem_type_map.end() && (iter->second == kMemTypeCPUInput || iter->second == kMemTypeCPUOutput);
}

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

  const MemTypeMap& InputMemoryType() const {
    return input_memory_type_args_;
  }

  const MemTypeMap& OutputMemoryType() const {
    return output_memory_type_args_;
  }

  // legacy interface for winml, should not be used in Lotus
  const MemTypeMap& MemoryType() const {
    return output_memory_type_args_;
  }

  int ExecQueueId() const {
    return exec_queue_id_;
  }

  bool IsConflict(const KernelDef& other) const {
    if (op_name_ != other.OpName() || provider_type_ != other.Provider())
      return false;
    int start = 0, end = 0;
    other.SinceVersion(&start, &end);
    if (op_since_version_start_ > end || op_since_version_end_ < start)
      return false;
    //check types
    auto other_types = other.TypeConstraints();
    bool type_conflict = false;
    for (auto it : type_constraints_) {
      if (other_types.count(it.first)) {
        for (auto type : it.second) {
          if (std::find(other_types[it.first].begin(), other_types[it.first].end(), type) != other_types[it.first].end())
            type_conflict = true;
        }
      }
    }
    if (!type_conflict)
      return false;
    //if has type conflict, check if any other field has different
    //for example, we register two kernel with float type, but one is inplace, another is not.
    //check in-place
    if (inplace_map_.empty() && !other.MayInplace().empty())
      return false;
    for (auto& it : inplace_map_) {
      if (std::find(other.MayInplace().begin(), other.MayInplace().end(), it) == other.MayInplace().end())
        return false;
    }

    //check alias
    for (auto& it : alias_map_) {
      if (std::find(other.Alias().begin(), other.Alias().end(), it) == other.Alias().end())
        return false;
    }
    if (alias_map_.empty() && !other.Alias().empty())
      return false;

    //check memory type
    auto other_input_mem_types = other.InputMemoryType();
    for (auto it : input_memory_type_args_) {
      if (other_input_mem_types.count(it.first) && other_input_mem_types[it.first] == it.second)
        return false;
    }
    if (input_memory_type_args_.empty() && !other.InputMemoryType().empty())
      return false;

    auto other_output_mem_types = other.OutputMemoryType();
    for (auto it : output_memory_type_args_) {
      if (other_output_mem_types.count(it.first) && other_output_mem_types[it.first] == it.second)
        return false;
    }
    if (output_memory_type_args_.empty() && !other.OutputMemoryType().empty())
      return false;

    return true;
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
  MemTypeMap input_memory_type_args_;
  MemTypeMap output_memory_type_args_;

  // execution command queue id, 0 for default queue in execution provider
  int exec_queue_id_ = 0;
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

  // allowing output[output_index] to reuse memory of input[input_index]
  KernelDefBuilder& MayInplace(int input_index, int output_index) {
    // TODO: validate inputs.
    kernel_def_->inplace_map_.emplace_back(input_index, output_index);
    return *this;
  }

  // Alias mapping from inputs to outputs. Different from Inplace that the
  // content of the tensor is not changed. This is to take care of operators
  // such as Identity and Reshape.
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    kernel_def_->alias_map_ = aliases;
    return *this;
  }

  KernelDefBuilder& Alias(int input_index, int output_index) {
    kernel_def_->alias_map_.emplace_back(input_index, output_index);
    return *this;
  }

  // Specify that this kernel requires an input arg
  // in certain memory type (instead of the default, device memory).
  template <MemType T>
  KernelDefBuilder& InputMemoryType(int input_index) {
    kernel_def_->input_memory_type_args_.insert(std::make_pair(input_index, T));
    return *this;
  }

  // Specify that this kernel provides an output arg
  // in certain memory type (instead of the default, device memory).
  template <MemType T>
  KernelDefBuilder& OutputMemoryType(int output_index) {
    kernel_def_->output_memory_type_args_.insert(std::make_pair(output_index, T));
    return *this;
  }

  // legacy interface for winml, should not be used in Lotus
  template <MemType T>
  KernelDefBuilder& MemoryType(int output_index) {
    kernel_def_->output_memory_type_args_.insert(std::make_pair(output_index, T));
    return *this;
  }

  // Specify that this kernel runs on which execution queue in the provider
  KernelDefBuilder& ExecQueueId(int queue_id) {
    kernel_def_->exec_queue_id_ = queue_id;
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
