#include "core/platform/windows/logging/etw_sink.h"

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED

#include "core/common/logging/capture.h"
#include "core/common/logging/logging.h"

#include "test/common/logging/helpers.h"

using namespace Lotus::Logging;

/// <summary>
/// Test usage of the ETW sinks does not fail.
/// </summary>
TEST(LoggingTests, TestEtwSink) {
  const std::string logid{"ETW"};
  const std::string message{"Test message"};

  // create scoped manager so sink gets destroyed once done and we check disposal
  // within the scope of this test
  {
    LoggingManager manager{std::unique_ptr<ISink>{new EtwSink{}}, Severity::kWARNING, false, "default"};

    auto logger = manager.CreateLogger(logid);

    LOGS(logger.get(), WARNING, Category::Lotus) << message;

    // can't test much else without creating an interface for ETW, using that in EtwSink
    // and mocking that interface here. too much work given how trivial the logic in EtwSink is.
  }
}

/// <summary>
/// Test that attempting to create two ETW sinks fails
/// </summary>
TEST(LoggingTests, TestEtwSinkCtor) {
  EtwSink sink1{};

  EXPECT_THROW(EtwSink sink2, std::logic_error);
}

#endif  // LOTUS_ETW_TRACE_LOGGING_SUPPORTED
