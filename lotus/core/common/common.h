/**
* Derived from caffe2, need copy right annoucement here.
*/

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/common/status.h"
#include "core/common/exceptions.h"

namespace Lotus {

template <typename Key, typename Value>
using LotusMap = std::unordered_map<Key, Value>;

// Using statements for common classes that we refer to in lotus very often.
// TODO(Task:137) Remove 'using' statements from header files
using std::set;
using std::string;
using std::unique_ptr;
using std::vector;

using Common::Status;
using Common::StatusCategory;
using Common::StatusCode;

#define UNUSED_PARAMETER(x) (x)

#define LOTUS_THROW(...)        \
  throw ::Lotus::EnforceNotMet( \
      __FILE__, __LINE__, "", ::Lotus::MakeString(__VA_ARGS__))

// Just in order to mark things as not implemented. Do not use in final code.
#define LOTUS_NOT_IMPLEMENTED LOTUS_THROW("Not Implemented.")

#define LOTUS_ENFORCE(condition, ...)                                        \
  do {                                                                       \
    if (!(condition)) {                                                      \
      throw ::Lotus::EnforceNotMet(                                          \
          __FILE__, __LINE__, #condition, ::Lotus::MakeString(__VA_ARGS__)); \
    }                                                                        \
  } while (false)

// Macros to disable the copy and/or move ctor and assignment methods
// These are usually placed in the private: declarations for a class.

#define LOTUS_DISALLOW_COPY(TypeName) \
  TypeName(const TypeName&) = delete

#define LOTUS_DISALLOW_ASSIGN(TypeName) \
  TypeName& operator=(const TypeName&) = delete

#define LOTUS_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  LOTUS_DISALLOW_COPY(TypeName);                 \
  LOTUS_DISALLOW_ASSIGN(TypeName)

#define LOTUS_DISALLOW_MOVE(TypeName)  \
  TypeName(const TypeName&&) = delete; \
  TypeName& operator=(const TypeName&&) = delete

#define LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TypeName) \
  LOTUS_DISALLOW_COPY_AND_ASSIGN(TypeName);           \
  LOTUS_DISALLOW_MOVE(TypeName)

#define LOTUS_RETURN_IF_ERROR(expr)        \
  do {                                     \
    auto _status = (expr);                 \
    if ((!_status.IsOK())) return _status; \
  } while (0)

#if defined(__GNUC__)
#if __GNUC_PREREQ(4, 9)
#define LOTUS_EXPORT [[gnu::visibility("default")]]
#else
#define LOTUS_EXPORT __attribute__((__visibility__("default")))
#endif
#else
#define LOTUS_EXPORT
#endif

// make_unique is a C++14 feature. If we don't have 14, we will emulate
// its behavior. This is copied from folly/Memory.h
#if __cplusplus >= 201402L ||                                              \
    (defined __cpp_lib_make_unique && __cpp_lib_make_unique >= 201304L) || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
/* using override */
using std::make_unique;
#else

template <typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T> >::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Allows 'make_unique<T[]>(10)'. (N3690 s20.9.1.4 p3-4)
template <typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T> >::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// Disallows 'make_unique<T[10]>()'. (N3690 s20.9.1.4 p5)
template <typename T, typename... Args>
typename std::enable_if<
    std::extent<T>::value != 0, std::unique_ptr<T> >::type
make_unique(Args&&...) = delete;

#endif

inline std::string StripBasename(const std::string& full_path) {
  const char kSeparator = '/';
  size_t pos = full_path.rfind(kSeparator);
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

inline void MakeStringInternal(std::stringstream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void
MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline string MakeString(const string& str) {
  return str;
}
inline string MakeString(const char* p_str) {
  return string(p_str);
}

}  // namespace Lotus
