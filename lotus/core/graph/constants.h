#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"

namespace LotusIR {
static const std::string kNoOp = "NoOp";
static const std::string kConstant = "Constant";
static const std::string kConstantValue = "value";
static const std::string kOnnxDomain = "";
static const std::string kMLDomain = "ai.onnx.ml";
static const std::string kCpuExecutionProvider = "CPUExecutionProvider";

// Singleton wrapper around allowed data types.
// This implements construct on first use which is needed to ensure
// static objects are initialized before use. Ops registration does not work
// properly without this.
class TypesWrapper {
 public:
  static TypesWrapper& GetTypesWrapper();

  // DataType strings. These should match the DataTypes defined in Data.proto
  const std::string kFloat16 = "float16";
  const std::string kFloat = "float";
  const std::string kDouble = "double";
  const std::string kInt8 = "int8";
  const std::string kInt16 = "int16";
  const std::string kInt32 = "int32";
  const std::string kInt64 = "int64";
  const std::string kUInt8 = "uint8";
  const std::string kUInt16 = "uint16";
  const std::string kUInt32 = "uint32";
  const std::string kUInt64 = "uint64";
  const std::string kComplex64 = "complex64";
  const std::string kComplex128 = "complex128";
  const std::string kString = "string";
  const std::string kBool = "bool";
  const std::string kUndefined = "undefined";

  const std::unordered_set<std::string>& GetAllowedDataTypes();

 private:
  TypesWrapper() = default;
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TypesWrapper);
};

// Singleton class used to help initialize static objects related to type strings.
// This is not strictly needed but allows common rich type strings to be defined here along
// side the data type strings above in TypesWrapper.
class TypeStringsInitializer {
 public:
  static TypeStringsInitializer& Instance();

 private:
  TypeStringsInitializer();
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TypeStringsInitializer);

  // Common string representations of TypeProto. These are used to pre-initialize
  // typeStringToProto map. Note: some of these strings may have already been initialized in
  // the map via op registration depending on static initialization order.
  const std::vector<std::string> kCommonTypeStrings = {"tensor(float16)", "tensor(float)",
                                                       "tensor(double)", "tensor(int8)", "tensor(int16)", "tensor(int32)",
                                                       "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
                                                       "tensor(uint64)", "tensor(complex64)", "tensor(complex128)", "tensor(string)",
                                                       "tensor(bool)"};
};

}  // namespace LotusIR
