// @@COPYRIGHT@@
#include "core/common/log_sink.h"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <sstream>

namespace Lotus {
std::string MakeLogMessageHeader(
    Lotus::LogSeverity severity,
    const char* file, int line,
    const char* category,
    const ::tm* tm_time, int microseconds) {
  std::ostringstream os{};
  os << '['
     // category
     << category
     << ' '
     // severity
     << LOTUS_SEVERITY_PREFIX[std::min(4, LOTUS_LOG_SEVERITY_FATAL - severity)]
     << ' '
     // timestamp
     << std::setfill('0') << std::setw(4) << tm_time->tm_year + 1900 << '-'
     << std::setw(2) << tm_time->tm_mon + 1 << '-'
     << std::setw(2) << tm_time->tm_mday << ' '
     << std::setw(2) << tm_time->tm_hour << ':'
     << std::setw(2) << tm_time->tm_min << ':'
     << std::setw(2) << tm_time->tm_sec << '.'
     << std::setw(6) << microseconds
     << ' '
     // location
     << file << ':' << line
     << "] ";
  return os.str();
}

std::mutex StdErrLogSink::stderr_mutex_{};

void StdErrLogSink::send(Lotus::LogSeverity severity, const char* /*full_filename*/,
                         const char* base_filename, int line,
                         const struct ::tm* tm_time, int microseconds,
                         const char* category,
                         const char* message, size_t message_len) {
  std::lock_guard<std::mutex> guard{stderr_mutex_};
  std::cerr << MakeLogMessageHeader(severity, base_filename, line,
                                    category, tm_time, microseconds)
            << std::string{message, message_len} << '\n';
}

bool InitLogSink(int* /*pargc*/, char*** /*pargv*/) {
  Lotus::SetLogSink(GetDefaultLogSink());
  return true;
}
}  // namespace Lotus
