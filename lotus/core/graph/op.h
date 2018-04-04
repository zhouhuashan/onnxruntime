#pragma once

#include <functional>
#include <unordered_map>
#include "onnx/defs/schema.h"
#include "core/common/status.h"
#include "core/graph/constants.h"

using namespace onnx;
using namespace Lotus::Common;

namespace LotusIR {
typedef AttributeProto_AttributeType AttrType;
typedef std::unordered_map<std::string, AttributeProto> NodeAttributes;

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

class TypeUtils {
 public:
  // Get attribute type given attribute proto data.
  static Status GetType(const AttributeProto& attr, AttrType& type);
  static bool IsValidAttribute(const AttributeProto& attribute);
};

class MsOpRegistry {
 public:
  static Status RegisterMsOps() {
    RETURN_IF_ERROR(RegisterMsActivationOps());
    RETURN_IF_ERROR(RegisterMsNNOps());
    return Status::OK();
  }

 private:
  static Status RegisterMsActivationOps();
  static Status RegisterMsNNOps();
};

}  // namespace LotusIR
