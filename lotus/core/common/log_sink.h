// @@COPYRIGHT@@
#ifndef _LOTUS_CORE_PLATFORM_LOG_SINK_H_
#define _LOTUS_CORE_PLATFORM_LOG_SINK_H_

#include <mutex>

#include "core/common/logging.h"

namespace Lotus {
using LogSinkPtr = std::unique_ptr<LogSinkInterface>;

/**
    * Gets the default log sink.
    */
LogSinkPtr GetDefaultLogSink();

/**
    * Returns a header for a log message.
    */
std::string MakeLogMessageHeader(
    LogSeverity,
    const char* file, int line,
    const char* category,
    const ::tm* tm_time, int microseconds);

/**
    * A log sink that sends log messages to a number of other log sinks.
    */
class CompositeLogSink : public LogSinkInterface {
 public:
  using LogSinkSharedPtr = std::shared_ptr<LogSinkInterface>;
  void AddLogSink(LogSinkSharedPtr log_sink) {
    log_sinks_.emplace_back(log_sink);
  }

  virtual void send(LogSeverity severity, const char* full_filename,
                    const char* base_filename, int line,
                    const struct ::tm* tm_time, int microseconds,
                    const char* category,
                    const char* message, size_t message_len) override {
    for (auto& log_sink : log_sinks_) {
      log_sink->send(severity, full_filename, base_filename, line,
                     tm_time, microseconds, category, message, message_len);
    }
  }

 private:
  std::vector<LogSinkSharedPtr> log_sinks_;
};

/*
    * A log sink that writes to stderr.
    */
class StdErrLogSink : public LogSinkInterface {
 public:
  virtual void send(LogSeverity severity, const char* full_filename,
                    const char* base_filename, int line,
                    const struct ::tm* tm_time, int microseconds,
                    const char* category,
                    const char* message, size_t message_len) override;

 private:
  // synchronize access to stderr from all instances
  static std::mutex stderr_mutex_;
};
}  // namespace Lotus

#endif  // _LOTUS_CORE_PLATFORM_LOG_SINK_H_