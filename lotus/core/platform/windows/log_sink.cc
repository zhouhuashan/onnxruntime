// @@COPYRIGHT@@
#include "core/platform/log_sink.h"
#include "core/platform/windows/trace_logging_manager.h"
#include <Windows.h>

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED
#include <evntrace.h>
#include <TraceLoggingProvider.h>
#endif // LOTUS_ETW_TRACE_LOGGING_SUPPORTED

namespace Lotus {
    /**
    * A log sink that writes via OutputDebugString().
    */
    class DebugStringLogSink : public Lotus::LogSinkInterface
    {
    public:
        virtual void send(LogSeverity severity, const char* /*full_filename*/,
            const char* base_filename, int line,
            const struct ::tm* tm_time, int microseconds,
            const char* category,
            const char* message, size_t message_len) override
        {
            const std::string log_message =
                MakeLogMessageHeader(severity, base_filename, line, category,
                    tm_time, microseconds)
                + std::string{ message, message_len } + "\n";
            ::OutputDebugStringA(log_message.c_str());
        }
    };

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED
    /**
    * A log sink that writes using the ETW APIs.
    */
    class EtwLogSink : public LogSinkInterface
    {
    public:
        EtwLogSink()
        {
            // ensure the TraceLoggingManager is initialized
            TraceLoggingManager::Instance();
        }

        virtual void send(LogSeverity severity, const char* /*full_filename*/,
            const char* /*base_filename*/, int /*line*/,
            const struct ::tm* /*tm_time*/, int /*microseconds*/,
            const char* category,
            const char* message, size_t message_len) override
        {
            auto handle = TraceLoggingManager::Instance().Access();
            if (!handle.IsValid()) return;
            WriteTraceLog(handle.GetProvider(), severity, category,
                          message, static_cast<UINT16>(message_len));
        }

    private:
        static void WriteTraceLog(TraceLoggingManager::ProviderHandle provider,
                                  LogSeverity severity, const char* category,
                                  const char* message, UINT16 message_len) noexcept
        {
            if (severity < LOTUS_LOG_SEVERITY_INFO) severity = LOTUS_LOG_SEVERITY_INFO;
            else if (severity > LOTUS_LOG_SEVERITY_FATAL) severity = LOTUS_LOG_SEVERITY_FATAL;

            // the following macros accommodate the TraceLogging macros' need
            //   for string literals
#define EVENT_NAME "LotusRtLogEvent"

#define TRACELOGGINGWRITE_ARGS(level) \
    TraceLoggingLevel(level), \
    TraceLoggingString(category, "category"), \
    TraceLoggingCountedString(message, message_len, "message")

            // kind of clunky to call TraceLoggingWrite within each case, but
            //   TraceLoggingLevel requires a compile-time constant
            switch (severity)
            {
            case LOTUS_LOG_SEVERITY_INFO:
                TraceLoggingWrite(provider, EVENT_NAME,
                    TRACELOGGINGWRITE_ARGS(TRACE_LEVEL_INFORMATION));
                break;
            case LOTUS_LOG_SEVERITY_WARNING:
                TraceLoggingWrite(provider, EVENT_NAME,
                    TRACELOGGINGWRITE_ARGS(TRACE_LEVEL_WARNING));
                break;
            case LOTUS_LOG_SEVERITY_ERROR:
                TraceLoggingWrite(provider, EVENT_NAME,
                    TRACELOGGINGWRITE_ARGS(TRACE_LEVEL_ERROR));
                break;
            case LOTUS_LOG_SEVERITY_FATAL:
                TraceLoggingWrite(provider, EVENT_NAME,
                    TRACELOGGINGWRITE_ARGS(TRACE_LEVEL_CRITICAL));
                break;
            }
#undef EVENT_NAME
#undef TRACELOGGINGWRITE_ARGS
        }
    };
#endif // LOTUS_ETW_TRACE_LOGGING_SUPPORTED

    LogSinkPtr GetDefaultLogSink()
    {
        auto sink = std::make_unique<CompositeLogSink>();

#ifdef LOTUS_ENABLE_DEBUG_LOGGING
        // OutputDebugString sink
        sink->AddLogSink(std::make_unique<DebugStringLogSink>());
#endif

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED
        // ETW sink
        sink->AddLogSink(std::make_unique<EtwLogSink>());
#endif

#ifdef LOTUS_ENABLE_STDERR_LOGGING
        // stderr sink
        sink->AddLogSink(std::make_unique<StdErrLogSink>());
#endif

        return std::move(sink);
    }
} // namespace lotusrt