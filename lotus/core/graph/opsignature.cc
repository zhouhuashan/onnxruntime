#include "core/graph/opsignature.h"

namespace LotusIR {
OpSignature::FormalParameter::FormalParameter(const std::string& name, const std::string& type,
                                              const std::string& description,
                                              const TypeConstraintMap& constraint_map)
    : name_(name), type_str_(type), description_(description) {
  auto it = constraint_map.find(type);
  if (it != constraint_map.end()) {
    types_ = it->second.first;
  } else {
    if (!type.empty()) {
      types_.emplace(Utils::OpUtils::ToType(type_str_));
    }
  }
}

const std::string& OpSignature::FormalParameter::GetName() const {
  return name_;
}

const DataTypeSet& OpSignature::FormalParameter::GetTypes() const {
  return types_;
}

const std::string& OpSignature::FormalParameter::GetTypeStr() const {
  return type_str_;
}

const std::string& OpSignature::FormalParameter::GetDescription() const {
  return description_;
}

OpSignature::Attribute::Attribute(
    const std::string& attr_name,
    AttrType type,
    const std::string& description,
    const AttributeProto& default_value)
    : name_(attr_name), type_(type), description_(description), has_default_value_(true) {
  allowed_values_.push_back(default_value);
}

OpSignature::Attribute::Attribute(
    const std::string& attr_name,
    AttrType type,
    const std::string& description)
    : name_(attr_name), type_(type), description_(description), has_default_value_(false) {
}

const std::string& OpSignature::Attribute::GetName() const {
  return name_;
}

AttrType OpSignature::Attribute::GetType() const {
  return type_;
}

bool OpSignature::Attribute::HasDefaultValue(
    const AttributeProto** pp_value) const {
  if (has_default_value_ && nullptr != pp_value) {
    *pp_value = &(allowed_values_[0]);
  }

  return has_default_value_;
}

const std::string& OpSignature::GetName() const {
  return name_;
}

int OpSignature::SinceVersion() const {
  return since_version_;
}

const std::string& OpSignature::Domain() const {
  return domain_;
}

const std::string& OpSignature::GetDescription() const {
  return description_;
}

const std::vector<OpSignature::FormalParameter>&
OpSignature::GetInputs() const {
  return inputs_;
}

const std::vector<OpSignature::FormalParameter>&
OpSignature::GetOutputs() const {
  return outputs_;
}

const std::vector<OpSignature::Attribute>&
OpSignature::GetAttributes() const {
  return attributes_;
}

const TypeConstraintMap& OpSignature::GetTypeConstraintMap() const {
  return type_constraint_map_;
}

bool OpSignature::IsValidAttribute(const AttributeProto& attr) {
  if (attr.name().empty()) {
    return false;
  }

  if (attr.type() == AttributeProto_AttributeType_UNDEFINED) {
    int num_fields =
        attr.has_f() +
        attr.has_i() +
        attr.has_s() +
        attr.has_t() +
        attr.has_g() +
        (attr.floats_size() > 0) +
        (attr.ints_size() > 0) +
        (attr.strings_size() > 0) +
        (attr.tensors_size() > 0) +
        (attr.graphs_size() > 0);

    if (num_fields != 1) {
      return false;
    }
  }
  return true;
}
}  // namespace LotusIR
