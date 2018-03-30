#include <exception>

#include "core/common/exceptions.h"
#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"

namespace Lotus {
namespace Logging {
const char *Category::Lotus = "Lotus";
const char *Category::System = "System";

using namespace std::chrono;

/*
As LoggingManager can be a static, we need to wrap the default instance and mutex in functions
to ensure they're initialized before use in LoggingManager::LoggingManager. If we don't, and
a static LoggingManager is created at startup, the file scope statics here may not have been
initialized.
*/

static std::atomic<void *> &DefaultLoggerManagerInstance() {
  // this atomic is to protect against attempts to log being made after the default LoggingManager is destroyed.
  // Theoretically this can happen if a Logger instance is still alive and calls Log via its internal
  // pointer to the LoggingManager.
  // As the first thing LoggingManager::Log does is check the static DefaultLoggerManagerInstance() is not null,
  // any further damage should be prevented (in theory).
  static std::atomic<void *> default_instance(nullptr);
  return default_instance;
}

static std::mutex &DefaultLoggerMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<Logger> LoggingManager::default_logger_;

static minutes InitLocaltimeOffset(const time_point<system_clock> &epoch);

// we save the value from system clock (which we can convert to a timestamp) as well as the high_resolution_clock.
// from then on, we use the delta from the high_resolution_clock and apply that to the
// system clock value.
const time_point<high_resolution_clock> LoggingManager::high_res_epoch_{high_resolution_clock::now()};
const time_point<system_clock> LoggingManager::system_epoch_{system_clock::now()};
const minutes LoggingManager::localtime_offset_from_utc_{InitLocaltimeOffset(LoggingManager::system_epoch_)};

LoggingManager::LoggingManager(std::unique_ptr<ISink> sink, Severity min_severity, bool filter_user_data,
                               const InstanceType instance_type, const std::string *default_logger_id)
    : sink_{std::move(sink)},
      min_severity_{min_severity},
      filter_user_data_{filter_user_data},
      owns_default_logger_{false} {
  if (!sink_) {
    throw std::logic_error("ISink must be provided.");
  }

  if (instance_type == InstanceType::Default) {
    if (default_logger_id == nullptr) {
      throw std::logic_error("default_logger_id must be provided if instance_type is InstanceType::Default");
    }

    // lock mutex to create instance, and enable logging
    // this matches the mutex usage in Shutdown
    std::lock_guard<std::mutex> guard(DefaultLoggerMutex());

    if (DefaultLoggerManagerInstance().load() != nullptr) {
      throw std::logic_error("Only one instance of LoggingManager created with InstanceType::Default can exist at any point in time.");
    }

    DefaultLoggerManagerInstance().store(this);

    // This assertion passes, so using the atomic to validate calls to Log should
    // be reasonably economical.
    // assert(DefaultLoggerManagerInstance().is_lock_free());

    CreateDefaultLogger(*default_logger_id);

    owns_default_logger_ = true;
  }
}

LoggingManager::~LoggingManager() {
  if (owns_default_logger_) {
    // lock mutex to reset DefaultLoggerManagerInstance() and free default logger from this instance.
    std::lock_guard<std::mutex> guard(DefaultLoggerMutex());

    DefaultLoggerManagerInstance().store(nullptr, std::memory_order::memory_order_release);

    default_logger_.reset();
  }
}

void LoggingManager::CreateDefaultLogger(const std::string &logger_id) {
  // only called from ctor in scope where DefaultLoggerMutex() is already locked

  if (default_logger_ != nullptr) {
    throw std::logic_error("Default logger already set. ");
  }

  default_logger_ = std::make_unique<Logger>(*this, logger_id);
}

std::unique_ptr<Logger> LoggingManager::CreateLogger(std::string logger_id) {
  auto logger = std::make_unique<Logger>(*this, logger_id);
  return logger;
}

void LoggingManager::Log(const std::string &logger_id, const Capture &message) const {
  // note: assumes check on OutputIsEnabled is done previously.
  // TODO: Is that good enough or do we need to double-check here
  sink_->Send(GetTimestamp(), logger_id, message);
}

static minutes InitLocaltimeOffset(const time_point<system_clock> &epoch) {
  // convert the system_clock time_point (UTC) to localtime and gmtime to calculate the difference.
  // we do this once, and apply that difference in GetTimestamp().
  // NOTE: If we happened to be running over a period where the time changed (e.g. daylight saving started)
  // we won't pickup the change. Not worth the extra cost to be 100% accurate 100% of the time.

  const time_t system_time_t = system_clock::to_time_t(epoch);
  tm local_tm;
  tm utc_tm;

#ifdef WIN32
  localtime_s(&local_tm, &system_time_t);
  gmtime_s(&utc_tm, &system_time_t);
#else
  localtime_r(&system_time_t, &local_tm);
  gmtime_r(&system_time_t, &utc_tm);
#endif

  double seconds = difftime(mktime(&local_tm), mktime(&utc_tm));

  // minutes should be accurate enough for timezone conversion
  return minutes{static_cast<int64_t>(seconds / 60)};
}

std::exception LoggingManager::LogFatalAndCreateException(const char *category,
                                                          const CodeLocation &location,
                                                          const char *format_str, ...) {
  std::string exception_msg;

  // create Capture in separate scope so it gets destructed (leading to log output) before we throw.
  {
    Lotus::Logging::Capture c{Lotus::Logging::LoggingManager::DefaultLogger(),
                              Lotus::Logging::Severity::kFATAL, category, Lotus::Logging::DataType::SYSTEM, location};
    va_list args;
    va_start(args, format_str);

    c.ProcessPrintf(format_str, args);
    va_end(args);

    exception_msg = c.Message();
  }

  return LotusException(location, exception_msg);
}
}  // namespace Logging
}  // namespace Lotus
