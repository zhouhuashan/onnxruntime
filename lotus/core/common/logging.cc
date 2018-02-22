/**
* Derived from caffe2, need copy right annoucement here.
*/
#include "core/common/logging.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>

namespace Lotus {

size_t ReplaceAll(std::string& s, const char* from, const char* to) {
  size_t numReplaced = 0;
  std::string::size_type lenFrom = std::strlen(from);
  std::string::size_type lenTo = std::strlen(to);
  for (std::string::size_type pos = s.find(from); pos != std::string::npos;
       pos = s.find(from, pos + lenTo)) {
    s.replace(pos, lenFrom, to);
    numReplaced++;
  }
  return numReplaced;
}

std::shared_ptr<LogSinkInterface> SetLogSink(
    std::shared_ptr<LogSinkInterface> logSink) {
  return MessageLogger::SetLogSink(logSink);
}

std::shared_ptr<LogSinkInterface> MessageLogger::sink_{};

static const char* const kDefaultLogCategory = "Lotus";

MessageLogger::MessageLogger(const char *file, int line, int severity)
  : MessageLogger{ file, line, severity, kDefaultLogCategory }
{
}

MessageLogger::MessageLogger(const char *file, int line, int severity,
                             const char *category)
  : severity_{ severity }, file_{ file }, line_{ line },
    category_{ category ? category : "" } {
  tag_ = "";
  time_ = Clock::now();
}

// Output the contents of the stream to the proper channel on destruction.
MessageLogger::~MessageLogger() {
  const std::string message = stream_.str();
  // log to sink
  auto sink = std::atomic_load(&sink_);
  if (sink) {
      tm t{};
      const time_t tt = Clock::to_time_t(time_);
      const auto microseconds = static_cast<int32_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              time_.time_since_epoch()).count() % 1000000);
#ifdef WIN32
      localtime_s(&t, &tt);
#else
      localtime_r(&tt, &t);
#endif
      sink->send(severity_, file_, StripBasename(file_).c_str(), line_,
          &t, microseconds, category_, message.c_str(), message.size());
  }
  if (severity_ == LOTUS_LOG_SEVERITY_FATAL) {
    DealWithFatal();
  }
  stream_ << "\n";
}

std::shared_ptr<LogSinkInterface> MessageLogger::SetLogSink(
    std::shared_ptr<LogSinkInterface> logSink) {
  return std::atomic_exchange(&sink_, logSink);
}
}  // namespace Lotus
