// EtwSink.h must come before the windows includes
#include "core/platform/windows/logging/etw_sink.h"

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED

// STL includes
#include <exception>

// ETW includes
// need space after Windows.h to prevent clang-format re-ordering breaking the build.
// TraceLoggingProvider.h must follow Windows.h
#include <Windows.h>

#include <TraceLoggingProvider.h>
#include <evntrace.h>

namespace Lotus {
namespace Logging {

namespace {

TRACELOGGING_DEFINE_PROVIDER(etw_provider_handle, "LotusTraceLoggingProvider",
                             // {929DD115-1ECB-4CB5-B060-EBD4983C421D}
                             (0x929dd115, 0x1ecb, 0x4cb5, 0xb0, 0x60, 0xeb, 0xd4, 0x98, 0x3c, 0x42, 0x1d));
}  // namespace

std::atomic_flag EtwSink::have_instance_ = ATOMIC_FLAG_INIT;

EtwSink::EtwSink() {
  // attempt to set to true, returning the current value
  if (EtwSink::have_instance_.test_and_set() == true) {
    // in use
    throw std::logic_error(
        "Attempt to create second EtwSink instance. "
        "Only one can be used at any point in time in order to manage the ETW registration correctly.");
  }

  const HRESULT etw_status = ::TraceLoggingRegister(etw_provider_handle);

  if (FAILED(etw_status)) {
    throw std::runtime_error("ETW registration failed. Logging will be broken: " + std::to_string(etw_status));
  }
}

void EtwSink::SendImpl(const Timestamp &timestamp, const std::string &logger_id, const Capture &message) {
  UNREFERENCED_PARAMETER(timestamp);

  // Do we want to output Verbose level messages via ETW at any point it time?
  // TODO: Validate if this filtering makes sense.
  if (message.Severity() <= Severity::kVERBOSE || message.DataType() == DataType::USER) {
    return;
  }

  // NOTE: Theoretically we could create an interface for all the ETW system interactions so we can separate
  // out those from the logic in this class so it is more testable.
  // Right now the logic is trivial, so that effort isn't worth it.

  // TraceLoggingWrite requires (painfully) a compile time constant for the TraceLoggingLevel,
  // forcing us to use an ugly macro for the call.
#define ETW_EVENT_NAME "LotusLogEvent"
#define TRACE_LOG_WRITE(level)                                                             \
  TraceLoggingWrite(etw_provider_handle, ETW_EVENT_NAME, TraceLoggingLevel(level),         \
                    TraceLoggingString(logger_id.c_str(), "logger"),                       \
                    TraceLoggingString(message.Category(), "category"),                    \
                    TraceLoggingString(message.Location().ToString().c_str(), "location"), \
                    TraceLoggingString(message.Message().c_str(), "message"))

  auto severity{message.Severity()};
  CHECK_NE(severity, Severity::kFATAL);

  switch (message.Severity()) {
    case Severity::kVERBOSE:
      TRACE_LOG_WRITE(TRACE_LEVEL_VERBOSE);
      break;
    case Severity::kINFO:
      TRACE_LOG_WRITE(TRACE_LEVEL_INFORMATION);
      break;
    case Severity::kWARNING:
      TRACE_LOG_WRITE(TRACE_LEVEL_WARNING);
      break;
    case Severity::kERROR:
      TRACE_LOG_WRITE(TRACE_LEVEL_ERROR);
      break;
    case Severity::kFATAL:
      TRACE_LOG_WRITE(TRACE_LEVEL_CRITICAL);
      break;
    default:
      throw std::logic_error("Unexpected Severity of " + static_cast<int>(message.Severity()));
  }

#undef ETW_EVENT_NAME
#undef TRACE_LOG_WRITE
}

EtwSink::~EtwSink() {
  ::TraceLoggingUnregister(etw_provider_handle);

  EtwSink::have_instance_.clear();
}
}  // namespace Logging
}  // namespace Lotus

#endif  // LOTUS_ETW_TRACE_LOGGING_SUPPORTED
