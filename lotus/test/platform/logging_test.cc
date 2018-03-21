#include "core/common/logging.h"
#include <gtest/gtest.h>
#include <algorithm>

namespace Lotus {

struct MyLogSink : public LogSinkInterface {
  using Messages = std::vector<std::pair<LogSeverity, std::string>>;
  virtual void send(LogSeverity severity, const char*,
                    const char*, int,
                    const struct ::tm*, int,
                    const char*,
                    const char* message, size_t message_len) override {
    messages.emplace_back(
        std::make_pair(severity, std::string{message, message_len}));
  }

  Messages messages{};
};

#ifndef NDEBUG
TEST(LoggingTest, TestLogSink) {
  auto logSink = std::make_shared<MyLogSink>();

  // log with test sink
  auto originalLogSink = SetLogSink(logSink);

  LOG(INFO) << "hello!";
  LOG(WARNING) << 3.14;

  ASSERT_EQ(logSink->messages.size(), 2);
  const MyLogSink::Messages expectedMessages{
      {LOTUS_LOG_SEVERITY_INFO, "hello!"},
      {LOTUS_LOG_SEVERITY_WARNING, "3.14"},
  };
  EXPECT_EQ(logSink->messages, expectedMessages);

  // log without test sink
  SetLogSink(originalLogSink);
  LOG(INFO) << 5;

  EXPECT_EQ(logSink->messages.size(), 2);
}
#endif

}  // namespace Lotus
