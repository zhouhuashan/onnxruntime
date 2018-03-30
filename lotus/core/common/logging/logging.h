#pragma once

#include <atomic>
#include <chrono>
#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "core/common/common.h"
#include "core/common/logging/capture.h"
#include "core/common/logging/severity.h"

#include "core/common/logging/macros.h"

/*

Logging overview and expected usage:

At program startup:
 * Create one or more ISink instances. If multiple, combine using composite_sink.
 * Create a LoggingManager instance with the sink/s with is_default_instance set to true
   * Only one instance should be created in this way, and it should remain valid for until the program 
     no longer needs to produce log output.

You can either use the static default Logger which LoggingManager will create when constructed
via LoggingManager::DefaultLogger(), or separate Logger instances each with different log ids
via LoggingManager::CreateLogger. 

The log id is passed to the ISink instance with the sink determining how the log id is used
in the output.

LoggingManager
 * creates the Logger instances used by the application
 * provides a static default logger instance
 * owns the log sink instance
 * applies checks on severity and output of user data

The log macros create a Capture instance to capture the information to log.
If the severity and/or user filtering settings would prevent logging, no evaluation
of the log arguments will occur, so no performance cost beyond the severity and user
filtering check.

A sink can do further filter as needed.

*/

namespace Lotus {
namespace Logging {

typedef std::chrono::time_point<std::chrono::system_clock> Timestamp;

#ifdef _DEBUG
static int max_vlog_level = 3;  // Set directly based on your needs.
#else
static const int max_vlog_level = INT_MIN;  // no VLOG output
#endif

enum class DataType {
  SYSTEM = 0,  ///< System data.
  USER = 1     ///< Contains potentially sensitive user data.
};

// Internal log categories.
// Logging interface takes const char* so arbitrary values can also be used.
struct Category {
  static const char *Lotus;   ///< General output
  static const char *System;  ///< Log output regarding interactions with the host system
                              // TODO: What other high level categories are meaningful? Model? Optimizer? Execution?
};

class ISink;
class Logger;
class Capture;

/// <summary>
/// The logging manager.
/// Owns the log sink and potentially provides a default Logger instance.
/// Provides filtering based on a minimum LogSeverity level, and of messages with DataType::User if enabled.
/// </summary>
class LoggingManager final {
 public:
  enum InstanceType {
    Default,  ///< Default instance of LoggingManager that should exist for the lifetime of the program
    Temporal  ///< Temporal instance. CreateLogger(...) should be used, however DefaultLogger() will NOT be provided via this instance.
  };

  /**
  Initializes a new instance of the LoggingManager class.
  @param sink The sink to write to. Use CompositeSink if you need to write to multiple places.
  @param min_severity The minimum severity. Attempts to log messages with lower severity will be ignored.
  @param filter_user_data If set to true ignore messages with DataType::User.
  @param instance_type If InstanceType::Default, this is the default instance of the LoggingManager 
  and is expected to exist for the lifetime of the program. 
  It creates and owns the default logger that calls to the static DefaultLogger method return.
  @param default_logger_id Logger Id to use for the default logger. nullptr/ignored if instance_type == Temporal.
  */
  LoggingManager(std::unique_ptr<ISink> sink, Severity min_severity, bool filter_user_data,
                 const InstanceType instance_type, const std::string *default_logger_id = nullptr);

  /**
  Creates a new logger instance which will use the provided logger_id. 
  @param logger_id The log identifier.
  @returns A new Logger instance that the caller owns.
  */
  std::unique_ptr<Logger> CreateLogger(std::string logger_id);

  /**
  Gets the default logger instance if set. Throws if no default logger is currently registered.
  @remarks
  Creating a LoggingManager instance with is_default_instance == true registers a default logger.
  Note that the default logger is only valid until the LoggerManager that registered it is destroyed.
  @returns The default logger if available.
  */
  static const Logger &DefaultLogger();

  /**
  Logs a FATAL level message and creates an exception that can be thrown with error information.
  @param category The log category.
  @param location The location the log message was generated.
  @param format_str The printf format string.
  @param ... The printf arguments.
  @returns A new Logger instance that the caller owns.
  */
  static std::exception LogFatalAndCreateException(const char *category,
                                                   const CodeLocation &location,
                                                   const char *format_str, ...);

  /**
  Check if output is enabled for the provided LogSeverity and DataType values.
  @param severity The severity.
  @param data_type Type of the data.
  @returns True if a message with these values will be logged.
  */
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept;

  /**
  Logs the message using the provided logger id.
  @param logger_id The log identifier.
  @param message The log message.
  */
  void Log(const std::string &logger_id, const Capture &message) const;

  ~LoggingManager();

 private:
  Timestamp GetTimestamp() const noexcept;
  void CreateDefaultLogger(const std::string &logger_id);

  const Severity min_severity_;
  const bool filter_user_data_;
  bool owns_default_logger_;

  std::unique_ptr<ISink> sink_;

  static std::unique_ptr<Logger> default_logger_;
  static const std::chrono::time_point<std::chrono::high_resolution_clock> high_res_epoch_;
  static const std::chrono::time_point<std::chrono::system_clock> system_epoch_;
  static const std::chrono::minutes localtime_offset_from_utc_;
};

/**
Logger provides a per-instance log id. Everything else is passed back up to the LoggingManager
*/
class Logger {
 public:
  /**
  Initializes a new instance of the Logger class.
  @param loggingManager The logging manager.
  @param id The identifier for messages coming from this Logger
  */
  Logger(const LoggingManager &loggingManager, std::string id)
      : logging_manager_{&loggingManager}, id_{id} {
  }

  /**
  Check if output is enabled for the provided LogSeverity and DataType values.
  @param severity The severity.
  @param data_type Type of the data.
  @returns True if a message with these values will be logged.
  */
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept {
    return logging_manager_->OutputIsEnabled(severity, data_type);
  }

  /**
  Logs the captured message.
  @param message The log message.
  */
  void Log(const Capture &message) const {
    logging_manager_->Log(id_, message);
  }

 private:
  const LoggingManager *logging_manager_;
  const std::string id_;
};

inline const Logger &LoggingManager::DefaultLogger() {
  const Logger *default_logger = default_logger_.get();
  if (default_logger == nullptr) {
    // fail early for attempted misuse. don't use logging macros as we have no logger.
    throw std::logic_error("Attempt to use DefaultLogger but none has been registered.");
  }

  return *default_logger;
}

inline bool LoggingManager::OutputIsEnabled(Severity severity, DataType data_type) const noexcept {
  return (severity >= min_severity_ && (data_type != DataType::USER || !filter_user_data_));
}

inline Timestamp LoggingManager::GetTimestamp() const noexcept {
  using namespace std::chrono;

  const auto high_res_now = high_resolution_clock::now();
  return time_point_cast<system_clock::duration>(system_epoch_ + (high_res_now - high_res_epoch_) + localtime_offset_from_utc_);
}

}  // namespace Logging
}  // namespace Lotus
