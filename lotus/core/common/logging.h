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

#ifndef LOTUS_CORE_PLATFORM_LOGGING_H_
#define LOTUS_CORE_PLATFORM_LOGGING_H_

#include <chrono>
#include <climits>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include "core/common/common.h"

// Log severity level constants.
constexpr int LOTUS_LOG_SEVERITY_FATAL = 3;
constexpr int LOTUS_LOG_SEVERITY_ERROR = 2;
constexpr int LOTUS_LOG_SEVERITY_WARNING = 1;
constexpr int LOTUS_LOG_SEVERITY_INFO = 0;
const char LOTUS_SEVERITY_PREFIX[] = "FEWIV";
// LOTUS_LOG_THRESHOLD is a compile time flag that would allow us to turn off
// logging at compile time so no logging message below that level is produced
// at all. The value should be between INT_MIN and CAFFE_FATAL.
#ifndef LOTUS_LOG_THRESHOLD
// If we have not defined the compile time log threshold, we keep all the
// log cases.
#define LOTUS_LOG_THRESHOLD INT_MIN
#endif  // LOTUS_LOG_THRESHOLD

namespace Lotus {

// ---------------------- Log Sink Interface ----------------------
// provide a LogSink interface similar to glog's
using LogSeverity = int;
class LogSinkInterface {
 public:
  virtual ~LogSinkInterface() = default;

  virtual void send(LogSeverity severity, const char* full_filename,
                    const char* base_filename, int line,
                    const struct ::tm* tm_time, int microseconds,
                    const char* category,
                    const char* message, size_t message_len) = 0;
};

class MessageLogger {
 public:
  MessageLogger(const char* file, int line, int severity);
  ~MessageLogger();
  // Return the stream associated with the logger object.
  std::stringstream& stream() { return stream_; }

  MessageLogger(const char* file, int line, int severity, const char* category);

  // Sets the log sink. Returns the previous log sink.
  static std::shared_ptr<LogSinkInterface> SetLogSink(
      std::shared_ptr<LogSinkInterface> logSink);

 private:
  // When there is a fatal log, we simply abort.
  void DealWithFatal() { abort(); }

  const char* tag_;
  std::stringstream stream_;
  int severity_;

  using Clock = std::chrono::system_clock;

  const char* file_;
  int line_;
  const char* category_;
  Clock::time_point time_;
  static std::shared_ptr<LogSinkInterface> sink_;
};

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LoggerVoidify {
 public:
  LoggerVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(const std::ostream& s) { (void)s; }
};

// Log a message and terminate.
template <class T>
void LogMessageFatal(const char* file, int line, const T& message) {
  MessageLogger(file, line, LOTUS_LOG_SEVERITY_FATAL).stream() << message;
}

// Helpers for CHECK_NOTNULL(). Two are necessary to support both raw pointers
// and smart pointers.
template <typename T>
T& CheckNotNullCommon(const char* file, int line, const char* names, T& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line, std::string(names));
  }
  return t;
}

template <typename T>
T* CheckNotNull(const char* file, int line, const char* names, T* t) {
  return CheckNotNullCommon(file, line, names, t);
}

template <typename T>
T& CheckNotNull(const char* file, int line, const char* names, T& t) {
  return CheckNotNullCommon(file, line, names, t);
}

// Forward declare these two, and define them after all the container streams
// operators so that we can recurse from pair -> container -> container -> pair
// properly.
template <class First, class Second>
std::ostream& operator<<(
    std::ostream& out, const std::pair<First, Second>& p);
template <class Iter>
void PrintSequence(std::ostream& ss, Iter begin, Iter end);

#define INSTANTIATE_FOR_CONTAINER(container)               \
  template <class... Types>                                \
  std::ostream& operator<<(                                \
      std::ostream& out, const container<Types...>& seq) { \
    PrintSequence(out, seq.begin(), seq.end());            \
    return out;                                            \
  }

INSTANTIATE_FOR_CONTAINER(std::vector)
INSTANTIATE_FOR_CONTAINER(std::map)
INSTANTIATE_FOR_CONTAINER(std::set)
#undef INSTANTIATE_FOR_CONTAINER

template <class First, class Second>
inline std::ostream& operator<<(
    std::ostream& out, const std::pair<First, Second>& p) {
  out << '(' << p.first << ", " << p.second << ')';
  return out;
}

template <class Iter>
inline void PrintSequence(std::ostream& out, Iter begin, Iter end) {
  // Output at most 100 elements -- appropriate if used for logging.
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0) out << ' ';
    out << *begin;
  }
  if (begin != end) {
    out << " ...";
  }
}

std::string StripBasename(const std::string& full_path);

// Replace all occurrences of "from" substring to "to" string.
// Returns number of replacements
size_t ReplaceAll(std::string& s, const char* from, const char* to);

// Sets the log sink to which log messages are sent. Returns the previous log
// sink.
std::shared_ptr<LogSinkInterface> SetLogSink(
    std::shared_ptr<LogSinkInterface> logSink);

}  // namespace Lotus

// ---------------------- Logging Macro definitions --------------------------
#define LOTUS_LOG_SEVERITY_NAME_TO_IDENTIFIER(severity_name) \
  (LOTUS_LOG_SEVERITY_##severity_name)

static_assert(LOTUS_LOG_THRESHOLD <= LOTUS_LOG_SEVERITY_FATAL,
              "LOTUS_LOG_THRESHOLD should at most be FATAL.");

// Note: The LOGN_* versions take a value or identifier for the log severity.
//       The LOG_* versions take the name portion of the log severity constant
//       identifiers. E.g., LOG(INFO) = LOGN(LOTUS_LOG_SEVERITY_INFO).

// If n is under the compile time caffe log threshold, LOGN(n) should not
// generate anything in optimized code.
#define LOGN(n)                 \
  if (n >= LOTUS_LOG_THRESHOLD) \
  ::Lotus::MessageLogger((char*)__FILE__, __LINE__, n).stream()

#define LOG(severity_name) \
  LOGN(LOTUS_LOG_SEVERITY_NAME_TO_IDENTIFIER(severity_name))

#define VLOG(n) LOGN((-n))

// Log at the specified level with the specified category.
#define LOGN_WITH_CATEGORY(n, category) \
  if (n >= LOTUS_LOG_THRESHOLD)         \
  ::Lotus::MessageLogger((char*)__FILE__, __LINE__, n, category).stream()

#define LOG_WITH_CATEGORY(severity_name, category)                         \
  LOGN_WITH_CATEGORY(LOTUS_LOG_SEVERITY_NAME_TO_IDENTIFIER(severity_name), \
                     category)

#define LOGN_IF(n, condition)                  \
  if (n >= LOTUS_LOG_THRESHOLD && (condition)) \
  ::Lotus::MessageLogger((char*)__FILE__, __LINE__, n).stream()

#define LOG_IF(severity_name, condition) \
  LOGN_IF(LOTUS_LOG_SEVERITY_NAME_TO_IDENTIFIER(severity_name), condition)

#define VLOG_IF(n, condition) LOGN_IF((-n), (condition))

// Log only if condition is not met. Otherwise evaluates to void.
#define FATAL_IF(condition) \
  condition ? (void)0 : ::Lotus::LoggerVoidify() & ::Lotus::MessageLogger((char*)__FILE__, __LINE__, LOTUS_LOG_SEVERITY_FATAL).stream()

// Check for a given boolean condition.
#define CHECK(condition) FATAL_IF(condition) \
                             << "Check failed: " #condition " "

#ifndef NDEBUG
// Debug only version of CHECK
#define DCHECK(condition) FATAL_IF(condition) \
                              << "Check failed: " #condition " "
#else
// Optimized version - generates no code.
#define DCHECK(condition) \
  if (false) CHECK(condition)
#endif  // NDEBUG

#define CHECK_OP(val1, val2, op) FATAL_IF((val1 op val2)) \
                                     << "Check failed: " #val1 " " #op " " #val2 " "

// Check_op macro definitions
#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)

#ifndef NDEBUG
// Debug only versions of CHECK_OP macros.
#define DCHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#else  // !NDEBUG
// These versions generate no code in optimized mode.
#define DCHECK_EQ(val1, val2) \
  if (false) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) \
  if (false) CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) \
  if (false) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) \
  if (false) CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) \
  if (false) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) \
  if (false) CHECK_OP(val1, val2, >)
#endif  // NDEBUG

// Check that a pointer is not null.
#define CHECK_NOTNULL(val) \
  ::Lotus::CheckNotNull(   \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// Debug only version of CHECK_NOTNULL
#define DCHECK_NOTNULL(val) \
  ::Lotus::CheckNotNull(    \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))
#else  // !NDEBUG
// Optimized version - generates no code.
#define DCHECK_NOTNULL(val) \
  if (false) CHECK_NOTNULL(val)
#endif  // NDEBUG

#endif  // LOTUS_CORE_PLATFORM_LOGGING_H_
