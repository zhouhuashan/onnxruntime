#include "core/framework/kernel_def_builder.h"

namespace Lotus {
bool KernelDef::IsConflict(const KernelDef& other) const {
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
}  // namespace Lotus
