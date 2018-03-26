#include "core/graph/constants.h"
#include "core/graph/utils.h"

namespace LotusIR {
TypesWrapper& TypesWrapper::GetTypesWrapper() {
  static TypesWrapper types;
  return types;
}

const std::unordered_set<std::string>& TypesWrapper::GetAllowedDataTypes() {
  static std::unordered_set<std::string> allowed_data_types = {
      kFloat16, kFloat, kDouble,
      kInt8, kInt16, kInt32, kInt64,
      kUInt8, kUInt16, kUInt32, kUInt64,
      kComplex64, kComplex128,
      kString, kBool};

  return allowed_data_types;
}

TypeStringsInitializer& TypeStringsInitializer::Instance() {
  static TypeStringsInitializer init_types;
  return init_types;
}

TypeStringsInitializer::TypeStringsInitializer() {
  // Initialize TypeStrToProtoMap using common type strings.
  for (const auto& t : kCommonTypeStrings) {
    Utils::OpUtils::ToType(t);
  }
}

// This ensures all static objects related to type strings get initialized.
// TypeStringsInitializer constructor populates TypeStrToProtoMap with common type strings.
// TypesWrapper() gets instantiated via call to OpUtils::FromString()
// which calls GetTypesWrapper().
// Note: due to non-deterministic static initialization order, some of the type strings
// may have already been added via Op Registrations which use those type strings.
static TypeStringsInitializer& type_strings_ = TypeStringsInitializer::Instance();
}  // namespace LotusIR
