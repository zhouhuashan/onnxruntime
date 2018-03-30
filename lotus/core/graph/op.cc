#include "core/graph/op.h"
#include <cstring>
#include "core/graph/constants.h"
#include "core/graph/opsignature.h"
#include "core/graph/utils.h"

namespace LotusIR {
const std::string& OperatorSchema::GetName() const {
  return op_signature_.GetName();
}

int OperatorSchema::SinceVersion() const {
  return op_signature_.SinceVersion();
}

const std::string& OperatorSchema::Domain() const {
  return op_signature_.Domain();
}

const OpSignature& OperatorSchema::GetOpSignature() const {
  return op_signature_;
}

ShapeInferenceFunc OperatorSchema::GetShapeInferenceFunc() const {
  return shape_inference_func_;
}

AttributeParser OperatorSchema::GetAttributeParser() const {
  return attr_parser_;
}

OperatorSchemaSetter&
OperatorSchemaSetter::Name(const std::string& op_name) {
  op_schema_.op_signature_.name_ = op_name;
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::SinceVersion(int op_set_version) {
  op_schema_.op_signature_.since_version_ = op_set_version;
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::SetDomain(const std::string& domain) {
  op_schema_.op_signature_.domain_ = domain;
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::Description(const std::string& description) {
  op_schema_.op_signature_.description_ = description;
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::Input(const std::string& input_name,
                            const std::string& description,
                            const std::string& type,
                            bool optional) /* TODO: add logic for this */
{
  UNUSED_PARAMETER(optional);
  inputs_.push_back(std::make_tuple(input_name, description, type));
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::Output(const std::string& output_name,
                             const std::string& description,
                             const std::string& type) {
  outputs_.push_back(std::make_tuple(output_name, description, type));
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::Attr(const std::string& attr_name,
                           const std::string& description,
                           AttrType attr_type, bool required) {
  UNUSED_PARAMETER(required);
  op_schema_.op_signature_.attributes_.push_back(
      OpSignature::Attribute(attr_name, attr_type, description));
  return *this;
}

#define ATTR_SETTER_BASIC_IMPL(type, field)                  \
  OperatorSchemaSetter&                                      \
  OperatorSchemaSetter::Attr(const std::string& attr_name,   \
                             const std::string& description, \
                             AttrType attr_type,             \
                             const type& default_value) {    \
    AttributeProto a;                                        \
    a.set_name(attr_name);                                   \
    a.set_##field(default_value);                            \
                                                             \
    op_schema_.op_signature_.attributes_.push_back(          \
        OpSignature::Attribute(attr_name,                    \
                               attr_type,                    \
                               description,                  \
                               a));                          \
                                                             \
    return *this;                                            \
  }

#define ATTR_SETTER_LIST_IMPL(type, field)                             \
  OperatorSchemaSetter&                                                \
  OperatorSchemaSetter::Attr(const std::string& attr_name,             \
                             const std::string& description,           \
                             AttrType attr_type,                       \
                             const std::vector<type>& default_value) { \
    AttributeProto a;                                                  \
    a.set_name(attr_name);                                             \
    for (const auto& v : default_value) {                              \
      a.add_##field(v);                                                \
    }                                                                  \
                                                                       \
    op_schema_.op_signature_.attributes_.push_back(                    \
        OpSignature::Attribute(attr_name,                              \
                               attr_type,                              \
                               description,                            \
                               a));                                    \
    return *this;                                                      \
  }

ATTR_SETTER_BASIC_IMPL(int64_t, i)
ATTR_SETTER_BASIC_IMPL(float, f)
ATTR_SETTER_BASIC_IMPL(std::string, s)
ATTR_SETTER_LIST_IMPL(int64_t, ints)
ATTR_SETTER_LIST_IMPL(float, floats)
ATTR_SETTER_LIST_IMPL(std::string, strings)

OperatorSchemaSetter&
OperatorSchemaSetter::TypeConstraint(const std::string& type_name,
                                     const std::vector<std::string>& constraints,
                                     const std::string& description) {
  constraints_.push_back(std::make_tuple(type_name, constraints, description));
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::SetShapeInferenceFunc(ShapeInferenceFunc shape_infer_func) {
  op_schema_.shape_inference_func_ = shape_infer_func;
  return *this;
}

OperatorSchemaSetter&
OperatorSchemaSetter::SetAttributeParser(AttributeParser attr_parser) {
  op_schema_.attr_parser_ = attr_parser;
  return *this;
}

OperatorSchemaSetter& OperatorSchemaSetter::FillUsing(std::function<void(OperatorSchemaSetter&)> populator) {
  if (populator) {
    populator(*this);
  }
  return *this;
}

Status TypeUtils::GetType(const AttributeProto& attr, AttrType& type) {
  if (!OpSignature::IsValidAttribute(attr)) {
    return Status(LOTUS, FAIL, "Invalid AttributeProto.");
  }

  type = attr.type();
  if (AttrType::AttributeProto_AttributeType_UNDEFINED == type) {
    if (attr.has_f()) {
      type = AttrType::AttributeProto_AttributeType_FLOAT;
    } else if (attr.has_i()) {
      type = AttrType::AttributeProto_AttributeType_INT;
    } else if (attr.has_s()) {
      type = AttrType::AttributeProto_AttributeType_STRING;
    } else if (attr.has_t()) {
      type = AttrType::AttributeProto_AttributeType_TENSOR;
    } else if (attr.has_g()) {
      type = AttrType::AttributeProto_AttributeType_GRAPH;
    } else if (attr.floats_size()) {
      type = AttrType::AttributeProto_AttributeType_FLOATS;
    } else if (attr.ints_size()) {
      type = AttrType::AttributeProto_AttributeType_INTS;
    } else if (attr.strings_size()) {
      type = AttrType::AttributeProto_AttributeType_STRINGS;
    } else if (attr.tensors_size()) {
      type = AttrType::AttributeProto_AttributeType_TENSORS;
    } else if (attr.graphs_size()) {
      type = AttrType::AttributeProto_AttributeType_GRAPHS;
    } else {
      return Status(LOTUS, FAIL, "Invalid AttributeProto.");
    }
  }
  return Status::OK();
}

OpSchemaRegistry::DomainToVersionRange::DomainToVersionRange() {
  // Increase the highest version when you make BC-breaking changes to the
  // operator schema on specific domain. Update the lowest version when it's
  // determined to remove too old version history.
  map_[kOnnxDomain] = std::make_pair(1, 2);
  map_[kMLDomain] = std::make_pair(1, 1);
}

const std::unordered_map<std::string, std::pair<int, int>>&
OpSchemaRegistry::DomainToVersionRange::Map() const {
  return map_;
}

OpSchemaRegistry::DomainToVersionRange& OpSchemaRegistry::DomainToVersionRange::Instance() {
  static DomainToVersionRange domain_to_version_range;
  return domain_to_version_range;
}

OpSchemaRegistry::OpSchemaRegisterOnce::OpSchemaRegisterOnce(OperatorSchemaSetter& op_schema_setter) {
  auto& op_schema = op_schema_setter.op_schema_;
  // Process type constraints.
  for (const auto& constraint : op_schema_setter.constraints_) {
    std::string name;
    std::vector<std::string> types;
    std::string desc;
    std::tie(name, types, desc) = constraint;

    auto it = op_schema.op_signature_.type_constraint_map_.find(name);
    LOTUS_ENFORCE(it == op_schema.op_signature_.type_constraint_map_.end(), "Constraint " + name + " already exists");

    DataTypeSet d;
    for (const auto& t : types) {
      d.insert(Utils::OpUtils::ToType(t));
    }

    op_schema.op_signature_.type_constraint_map_.insert(std::make_pair(name, std::make_pair(d, desc)));
  }

  op_schema.op_signature_.inputs_.reserve(op_schema_setter.inputs_.size());
  for (const auto& input : op_schema_setter.inputs_) {
    std::string name;
    std::string type;
    std::string desc;
    std::tie(name, desc, type) = input;
    op_schema.op_signature_.inputs_.push_back(
        OpSignature::FormalParameter(name, type, desc, op_schema.op_signature_.type_constraint_map_));
  }

  op_schema.op_signature_.outputs_.reserve(op_schema_setter.outputs_.size());
  for (const auto& output : op_schema_setter.outputs_) {
    std::string name;
    std::string type;
    std::string desc;
    std::tie(name, desc, type) = output;
    op_schema.op_signature_.outputs_.push_back(
        OpSignature::FormalParameter(name, type, desc,
                                     op_schema.op_signature_.type_constraint_map_));
  }

  auto& m = map_();
  auto& op_name = op_schema_setter.op_schema_.GetName();
  auto& op_domain = op_schema_setter.op_schema_.Domain();
  auto ver = op_schema_setter.op_schema_.SinceVersion();
  LOTUS_ENFORCE(m[op_name][op_domain].count(ver) == 0, "Entry exists for Op:", op_name, " Domain:", op_domain, " Version:", ver);
  m[op_name][op_domain].emplace(std::make_pair(ver, op_schema_setter.op_schema_));
}

const OperatorSchema* OpSchemaRegistry::Schema(const std::string& key,
                                               const std::string& domain) {
  auto& m = map_();
  if (m.count(key) && m[key].count(domain)) {
    return &m[key][domain].rbegin()->second;
  } else {
    return nullptr;
  }
}

const OperatorSchema* OpSchemaRegistry::Schema(const std::string& key,
                                               const int max_inclusive_version,
                                               const std::string& domain) {
  auto& m = map_();
  if (m.count(key) && m[key].count(domain)) {
    auto pos = m[key][domain].lower_bound(max_inclusive_version);
    if (m[key][domain].begin() == pos && pos->first > max_inclusive_version) {
      // All versions are greater than specified version.
      return nullptr;
    }

    if (m[key][domain].end() == pos || pos->first > max_inclusive_version) {
      // All versions are less than specified version, or,
      // The <pos> version is greater than specified version.
      pos--;
      return &(pos->second);
    }
    // Schema with exact version as specified one exists.
    return &(pos->second);
  } else {
    return nullptr;
  }
}

OpSchemaMap& OpSchemaRegistry::map_() {
  static OpSchemaMap map;
  return map;
}
}  // namespace LotusIR
