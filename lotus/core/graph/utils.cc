#include <cctype>
#include <iostream>
#include <iterator>
#include <sstream>

#include "core/graph/constants.h"
#include "core/graph/utils.h"
#include "core/protobuf/onnx-ml.pb.h"

using namespace onnx;

namespace LotusIR {
namespace Utils {
std::unordered_map<std::string, TypeProto>& OpUtils::GetTypeStrToProtoMap() {
  static std::unordered_map<std::string, TypeProto> map;
  return map;
}

std::mutex& OpUtils::GetTypeStrLock() {
  static std::mutex lock;
  return lock;
}

PTYPE OpUtils::ToType(const TypeProto& type) {
  auto type_str = ToString(type);
  std::lock_guard<std::mutex> lock(GetTypeStrLock());
  if (GetTypeStrToProtoMap().find(type_str) == GetTypeStrToProtoMap().end()) {
    GetTypeStrToProtoMap()[type_str] = type;
  }
  return &(GetTypeStrToProtoMap().find(type_str)->first);
}

PTYPE OpUtils::ToType(const std::string& type) {
  TypeProto type_proto;
  FromString(type, type_proto);
  return ToType(type_proto);
}

const TypeProto& OpUtils::ToTypeProto(const PTYPE& p_type) {
  std::lock_guard<std::mutex> lock(GetTypeStrLock());
  auto it = GetTypeStrToProtoMap().find(*p_type);
  LOTUS_ENFORCE(it != GetTypeStrToProtoMap().end(), "Type was not found: %s", *p_type);
  return it->second;
}

std::string OpUtils::ToString(const TypeProto& type, const std::string& left, const std::string& right) {
  switch (type.value_case()) {
    case TypeProto::ValueCase::kTensorType: {
      if (type.tensor_type().has_shape() && type.tensor_type().shape().dim_size() == 0) {
        // Scalar case.
        return left + ToDataTypeString(type.tensor_type().elem_type()) + right;
      } else {
        return left + "tensor(" + ToDataTypeString(type.tensor_type().elem_type()) + ")" + right;
      }
    }
    case TypeProto::ValueCase::kSequenceType:
      return ToString(type.sequence_type().elem_type(), left + "seq(", ")" + right);
    case TypeProto::ValueCase::kMapType: {
      std::string map_str = "map(" + ToDataTypeString(type.map_type().key_type()) + ",";
      return ToString(type.map_type().value_type(), left + map_str, ")" + right);
    }
    default:
      LOTUS_THROW("Invalid TypeProto.value_case(): %d", type.value_case());
  }
}

std::string OpUtils::ToDataTypeString(const TensorProto::DataType& type) {
  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  switch (type) {
    case TensorProto::DataType::TensorProto_DataType_BOOL:
      return t.kBool;
    case TensorProto::DataType::TensorProto_DataType_STRING:
      return t.kString;
    case TensorProto::DataType::TensorProto_DataType_FLOAT16:
      return t.kFloat16;
    case TensorProto::DataType::TensorProto_DataType_FLOAT:
      return t.kFloat;
    case TensorProto::DataType::TensorProto_DataType_DOUBLE:
      return t.kDouble;
    case TensorProto::DataType::TensorProto_DataType_INT8:
      return t.kInt8;
    case TensorProto::DataType::TensorProto_DataType_INT16:
      return t.kInt16;
    case TensorProto::DataType::TensorProto_DataType_INT32:
      return t.kInt32;
    case TensorProto::DataType::TensorProto_DataType_INT64:
      return t.kInt64;
    case TensorProto::DataType::TensorProto_DataType_UINT8:
      return t.kUInt8;
    case TensorProto::DataType::TensorProto_DataType_UINT16:
      return t.kUInt16;
    case TensorProto::DataType::TensorProto_DataType_UINT32:
      return t.kUInt32;
    case TensorProto::DataType::TensorProto_DataType_UINT64:
      return t.kUInt64;
    case TensorProto::DataType::TensorProto_DataType_COMPLEX64:
      return t.kComplex64;
    case TensorProto::DataType::TensorProto_DataType_COMPLEX128:
      return t.kComplex128;
    case TensorProto::DataType::TensorProto_DataType_UNDEFINED:
      return t.kUndefined;
  }

  LOTUS_THROW("Invalid TensorProto::DataType: %d", type);
}

void OpUtils::FromString(const std::string& src, TypeProto& type) {
  StringRange s(src);
  type.Clear();

  if (s.LStrip("seq")) {
    s.ParensWhitespaceStrip();
    return FromString(std::string(s.Data(), s.Size()), *(type.mutable_sequence_type()->mutable_elem_type()));
  } else if (s.LStrip("map")) {
    s.ParensWhitespaceStrip();
    size_t key_size = s.Find(',');
    StringRange k(s.Data(), key_size);
    std::string key = std::string(k.Data(), k.Size());
    s.LStrip(key_size);
    s.LStrip(",");
    StringRange v(s.Data(), s.Size());
    TensorProto::DataType key_type;
    FromDataTypeString(key, key_type);
    type.mutable_map_type()->set_key_type(key_type);
    return FromString(std::string(v.Data(), v.Size()), *type.mutable_map_type()->mutable_value_type());
  } else if (s.LStrip("tensor")) {
    s.ParensWhitespaceStrip();
    TensorProto::DataType e;
    FromDataTypeString(std::string(s.Data(), s.Size()), e);
    type.mutable_tensor_type()->set_elem_type(e);
  } else {
    // Scalar
    TensorProto::DataType e;
    FromDataTypeString(std::string(s.Data(), s.Size()), e);
    TypeProto::Tensor* t = type.mutable_tensor_type();
    t->set_elem_type(e);
    // Call mutable_shape() to initialize a shape with no dimension.
    t->mutable_shape();
  }
}

bool OpUtils::IsValidDataTypeString(const std::string& data_type) {
  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  const auto& allowed_set = t.GetAllowedDataTypes();
  return (allowed_set.find(data_type) != allowed_set.end());
}

void OpUtils::SplitStringTokens(StringRange& src, std::vector<StringRange>& tokens) {
  int parens = 0;
  src.RestartCapture();
  while (src.Size() > 0) {
    if (src.StartsWith(",")) {
      if (parens == 0) {
        tokens.push_back(src.GetCaptured());
        src.LStrip(",");
        src.RestartCapture();
      } else {
        src.LStrip(",");
      }
    } else if (src.LStrip("(")) {
      parens++;
    } else if (src.LStrip(")")) {
      parens--;
    } else {
      src.LStrip(1);
    }
  }
  tokens.push_back(src.GetCaptured());
}

void OpUtils::FromDataTypeString(const std::string& type_str, TensorProto::DataType& type) {
  LOTUS_ENFORCE(IsValidDataTypeString(type_str), "Invalid data type string of " + type_str);

  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  if (type_str == t.kBool) {
    type = TensorProto::DataType::TensorProto_DataType_BOOL;
  } else if (type_str == t.kFloat) {
    type = TensorProto::DataType::TensorProto_DataType_FLOAT;
  } else if (type_str == t.kFloat16) {
    type = TensorProto::DataType::TensorProto_DataType_FLOAT16;
  } else if (type_str == t.kDouble) {
    type = TensorProto::DataType::TensorProto_DataType_DOUBLE;
  } else if (type_str == t.kInt8) {
    type = TensorProto::DataType::TensorProto_DataType_INT8;
  } else if (type_str == t.kInt16) {
    type = TensorProto::DataType::TensorProto_DataType_INT16;
  } else if (type_str == t.kInt32) {
    type = TensorProto::DataType::TensorProto_DataType_INT32;
  } else if (type_str == t.kInt64) {
    type = TensorProto::DataType::TensorProto_DataType_INT64;
  } else if (type_str == t.kString) {
    type = TensorProto::DataType::TensorProto_DataType_STRING;
  } else if (type_str == t.kUInt8) {
    type = TensorProto::DataType::TensorProto_DataType_UINT8;
  } else if (type_str == t.kUInt16) {
    type = TensorProto::DataType::TensorProto_DataType_UINT16;
  } else if (type_str == t.kUInt32) {
    type = TensorProto::DataType::TensorProto_DataType_UINT32;
  } else if (type_str == t.kUInt64) {
    type = TensorProto::DataType::TensorProto_DataType_UINT64;
  } else if (type_str == t.kComplex64) {
    type = TensorProto::DataType::TensorProto_DataType_COMPLEX64;
  } else if (type_str == t.kComplex128) {
    type = TensorProto::DataType::TensorProto_DataType_COMPLEX128;
  } else {
    LOTUS_THROW("Invalid type string of " + type_str);
  }
}

StringRange::StringRange()
    : data_(""), size_(0), start_(data_), end_(data_) {}

StringRange::StringRange(const char* p_data, size_t size)
    : data_(p_data), size_(size), start_(data_), end_(data_) {
  LOTUS_ENFORCE(p_data != nullptr, "Null input");
  LAndRStrip();
}

StringRange::StringRange(const std::string& str)
    : data_(str.data()), size_(str.size()), start_(data_), end_(data_) {
  LAndRStrip();
}

StringRange::StringRange(const char* p_data)
    : data_(p_data), size_(strlen(p_data)), start_(data_), end_(data_) {
  LAndRStrip();
}

const char* StringRange::Data() const {
  return data_;
}

size_t StringRange::Size() const {
  return size_;
}

bool StringRange::Empty() const {
  return size_ == 0;
}

char StringRange::operator[](size_t idx) const {
  return data_[idx];
}

void StringRange::Reset() {
  data_ = "";
  size_ = 0;
  start_ = end_ = data_;
}

void StringRange::Reset(const char* p_data, size_t size) {
  data_ = p_data;
  size_ = size;
  start_ = end_ = data_;
}

void StringRange::Reset(const std::string& str) {
  data_ = str.data();
  size_ = str.size();
  start_ = end_ = data_;
}

bool StringRange::StartsWith(const StringRange& str) const {
  return ((size_ >= str.size_) && (memcmp(data_, str.data_, str.size_) == 0));
}

bool StringRange::EndsWith(const StringRange& str) const {
  return ((size_ >= str.size_) &&
          (memcmp(data_ + (size_ - str.size_), str.data_, str.size_) == 0));
}

bool StringRange::LStrip() {
  size_t count = 0;
  const char* ptr = data_;
  while (count < size_ && isspace(*ptr)) {
    count++;
    ptr++;
  }

  if (count > 0) {
    return LStrip(count);
  }
  return false;
}

bool StringRange::LStrip(size_t size) {
  if (size <= size_) {
    data_ += size;
    size_ -= size;
    end_ += size;
    return true;
  }
  return false;
}

bool StringRange::LStrip(StringRange str) {
  if (StartsWith(str)) {
    return LStrip(str.size_);
  }
  return false;
}

bool StringRange::RStrip() {
  size_t count = 0;
  const char* ptr = data_ + size_ - 1;
  while (count < size_ && isspace(*ptr)) {
    ++count;
    --ptr;
  }

  if (count > 0) {
    return RStrip(count);
  }
  return false;
}

bool StringRange::RStrip(size_t size) {
  if (size_ >= size) {
    size_ -= size;
    return true;
  }
  return false;
}

bool StringRange::RStrip(StringRange str) {
  if (EndsWith(str)) {
    return RStrip(str.size_);
  }
  return false;
}

bool StringRange::LAndRStrip() {
  bool l = LStrip();
  bool r = RStrip();
  return l || r;
}

void StringRange::ParensWhitespaceStrip() {
  LStrip();
  LStrip("(");
  LAndRStrip();
  RStrip(")");
  RStrip();
}

size_t StringRange::Find(const char ch) const {
  size_t idx = 0;
  while (idx < size_) {
    if (data_[idx] == ch) {
      return idx;
    }
    idx++;
  }
  return std::string::npos;
}

void StringRange::RestartCapture() {
  start_ = data_;
  end_ = data_;
}

StringRange StringRange::GetCaptured() {
  return StringRange(start_, end_ - start_);
}
}  // namespace Utils
}  // namespace LotusIR
