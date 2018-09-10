#pragma once

#include <ntverp.h>

// check for Windows 10 SDK or later
// https://stackoverflow.com/questions/2665755/how-can-i-determine-the-version-of-the-windows-sdk-installed-on-my-computer
#if VER_PRODUCTBUILD > 9600
// LotusRT ETW trace logging uses Windows 10 SDK's TraceLoggingProvider.h
#define LOTUS_ETW_TRACE_LOGGING_SUPPORTED 1
#endif

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED

#include <date/date.h>
#include <atomic>
#include <iostream>
#include <string>

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"

namespace onnxruntime {
namespace Logging {

class EtwSink : public ISink {
 public:
  EtwSink() = default;
  ~EtwSink() = default;

  constexpr static const char* kEventName = "LotusLogEvent";

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(EtwSink);

  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override;

  // limit to one instance of an EtwSink being around, so we can control the lifetime of
  // EtwTracingManager to ensure we cleanly unregister it
  static std::atomic_flag have_instance_;
};
}  // namespace Logging
}  // namespace onnxruntime

#endif  // LOTUS_ETW_TRACE_LOGGING_SUPPORTED
