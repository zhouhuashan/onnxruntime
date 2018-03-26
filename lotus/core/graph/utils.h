#pragma once

#include <fcntl.h>
#include <fstream>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/io.h>
#include <unistd.h>
#endif

#include <mutex>
#include <string>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/protobuf/onnx-ml.pb.h"

namespace {
using namespace ::Lotus::Common;
#ifdef _WIN32
inline Status FileOpenRd(const std::wstring& path, /*out*/ int* p_fd) {
  _wsopen_s(p_fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}

inline Status FileOpenWr(const std::wstring& path, /*out*/ int* p_fd) {
  _wsopen_s(p_fd, path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}
#endif

inline Status FileOpenRd(const std::string& path, /*out*/ int* p_fd) {
#ifdef _WIN32
  _sopen_s(p_fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
  *p_fd = open(path.c_str(), O_RDONLY);
#endif
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}

inline Status FileOpenWr(const std::string& path, /*out*/ int* p_fd) {
#ifdef _WIN32
  _sopen_s(p_fd, path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
  *p_fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}

inline Status FileClose(int fd) {
  int ret = 0;
#ifdef _WIN32
  ret = _close(fd);
#else
  ret = close(fd);
#endif
  if (0 != ret) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}
}  // namespace

namespace LotusIR {
typedef const std::string* PTYPE;  // TODO(Task:136): Clarify naming, definition and usage of Lotus::PTYPE
namespace Utils {
class StringRange;

class OpUtils {
 public:
  static PTYPE ToType(const onnx::TypeProto& type);
  static PTYPE ToType(const std::string& type);
  static const onnx::TypeProto& ToTypeProto(const PTYPE& p_type);
  static std::string ToString(const onnx::TypeProto& type, const std::string& left = "", const std::string& right = "");
  static std::string ToDataTypeString(const onnx::TensorProto::DataType& type);
  static void FromString(const std::string& src, onnx::TypeProto& type);
  static void FromDataTypeString(const std::string& src, onnx::TensorProto::DataType& type);
  static bool IsValidDataTypeString(const std::string& data_type);
  static void SplitStringTokens(StringRange& src, std::vector<StringRange>& tokens);

 private:
  static std::unordered_map<std::string, onnx::TypeProto>& GetTypeStrToProtoMap();
  // Returns lock used for concurrent updates to TypeStrToProtoMap.
  static std::mutex& GetTypeStrLock();
};

// Simple class which contains pointers to external string buffer and a size.
// This can be used to track a "valid" range/slice of the string.
// Caller should ensure StringRange is not used after external storage has
// been freed.
class StringRange {
 public:
  StringRange();
  StringRange(const char* p_data, size_t size);
  StringRange(const std::string& str);
  StringRange(const char* p_data);
  const char* Data() const;
  size_t Size() const;
  bool Empty() const;
  char operator[](size_t idx) const;
  void Reset();
  void Reset(const char* p_data, size_t size);
  void Reset(const std::string& str);
  bool StartsWith(const StringRange& str) const;
  bool EndsWith(const StringRange& str) const;
  bool LStrip();
  bool LStrip(size_t size);
  bool LStrip(StringRange str);
  bool RStrip();
  bool RStrip(size_t size);
  bool RStrip(StringRange str);
  bool LAndRStrip();
  void ParensWhitespaceStrip();
  size_t Find(const char ch) const;

  // These methods provide a way to return the range of the string
  // which was discarded by LStrip(). i.e. We capture the string
  // range which was discarded.
  StringRange GetCaptured();
  void RestartCapture();

 private:
  // data_ + size tracks the "valid" range of the external string buffer.
  const char* data_;
  size_t size_;

  // start_ and end_ track the captured range.
  // end_ advances when LStrip() is called.
  const char* start_;
  const char* end_;
};

// Use this to avoid compiler warnings about unused variables. E.g., if
// a variable is only used in an assert when compiling in Release mode.
// Adapted from https://stackoverflow.com/questions/15763937/unused-parameter-in-c11
template <typename... Args>
void Ignore(Args&&...) {}
}  // namespace Utils
}  // namespace LotusIR
