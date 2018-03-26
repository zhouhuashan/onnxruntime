#pragma once

#include <functional>
#include <unordered_map>

#include "core/graph/utils.h"
#include "core/protobuf/onnx-ml.pb.h"

using namespace onnx;

namespace LotusIR {
typedef AttributeProto_AttributeType AttrType;

// This string array should exactly match the AttrType defined above.
static const std::string kAttrTypeStrings[14] =
    {
        "FLOAT",
        "INT",
        "STRING",
        "GRAPH",
        "TENSOR",
        "FLOATS",
        "INTS",
        "STRINGS",
        "GRAPHS",
        "TENSORS"};

typedef std::unordered_set<PTYPE> DataTypeSet;
typedef std::unordered_map<std::string, std::pair<DataTypeSet, std::string>> TypeConstraintMap;

// Operator signature declaration.
// It defines input formal parameter, output formal parameters and
// attributes.
// Once an operator signature created, it's "Read-Only".
class OpSignature {
 public:
  // Formal parameter representation, including parameter name, type.
  class FormalParameter {
   public:
    // Constructor.
    explicit FormalParameter(const std::string& name,
                             const std::string& type,
                             const std::string& description,
                             const TypeConstraintMap& constraint_map = TypeConstraintMap());

    // Get formal parameter name.
    const std::string& GetName() const;

    // Get supported data types.
    const DataTypeSet& GetTypes() const;

    // Get formal parameter type string.
    const std::string& GetTypeStr() const;

    // Get formal parameter description.
    const std::string& GetDescription() const;

   private:
    FormalParameter() {}

    // Formal parameter name.
    std::string name_;

    // A set of data types supported for <*this> formal parameter.
    // It should contain at least one element if this formal parameter
    // is good.
    DataTypeSet types_;

    // The <parameter type> string specified when registering an op.
    // It could be a supported data type or a type constraint key, which
    // maps to a set of supported data types.
    std::string type_str_;

    // Formal parameter description
    std::string description_;
  };

  // Attribute representation, including name, type, and allowed values.
  // The first element of allowed values (if specified) is the default
  // value.
  class Attribute {
   public:
    // Constructor.
    explicit Attribute(const std::string& attr_name,
                       AttrType type,
                       const std::string& description);

    // Constructor with default value.
    explicit Attribute(const std::string& attr_name,
                       AttrType type,
                       const std::string& description,
                       const AttributeProto& default_value);

    // Get attribute name.
    const std::string& GetName() const;

    // Get attribute type.
    AttrType GetType() const;

    // Get to know whether this attribute has default value,
    // if yes, <p_value> will be assigned to be the default value.
    bool HasDefaultValue(const AttributeProto** pp_value) const;

   private:
    Attribute() {}

    // Attribute name.
    std::string name_;

    // Attribute type.
    AttrType type_;

    // Attribute description.
    std::string description_;

    // Flag indicates whether a default value specified.
    // It it's true, the first element of <allowed_values_> is the
    // default value.
    bool has_default_value_;

    // Allowed attribute values.
    std::vector<AttributeProto> allowed_values_;
  };

  static bool IsValidAttribute(const AttributeProto& attribute);

  // Constructor.
  OpSignature() = default;

  // Get operator name.
  const std::string& GetName() const;
  int SinceVersion() const;
  const std::string& Domain() const;

  // Get operator description.
  const std::string& GetDescription() const;

  // Get input formal parameters.
  const std::vector<FormalParameter>& GetInputs() const;

  // Get output formal parameters.
  const std::vector<FormalParameter>& GetOutputs() const;

  // Get attributes.
  const std::vector<Attribute>& GetAttributes() const;

  // Get type constraint map.
  const TypeConstraintMap& GetTypeConstraintMap() const;

 private:
  friend class OperatorSchemaSetter;
  friend class OpSchemaRegistry;

  // Operator name.
  std::string name_;
  int since_version_ = 1;
  std::string domain_ = "";

  // Operator description.
  std::string description_;

  // Operator input formal parameters.
  std::vector<FormalParameter> inputs_;

  // Operator output formal parameters.
  std::vector<FormalParameter> outputs_;

  // Operator attributes' definitions.
  std::vector<Attribute> attributes_;

  // Map from constraint name to DataTypeSet
  TypeConstraintMap type_constraint_map_;
};
}  // namespace LotusIR
