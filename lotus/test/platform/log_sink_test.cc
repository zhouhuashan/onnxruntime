#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/log_sink.h"

namespace Lotus {
namespace test {
    struct TestLogSink : public LogSinkInterface
    {
        virtual void send(LogSeverity, const char*,
            const char*, int,
            const struct ::tm*, int,
            const char*,
            const char* message, size_t message_len) override
        {
            messages_.emplace_back(message, message_len);
        }

        std::vector<std::string> messages_;
    };

    void SendLogMessage(LogSinkInterface& sink, LogSeverity severity,
                        const std::string& message)
    {
        static const ::tm t{};
        sink.send(severity, "", "", 0, &t, 0, "test_category", message.c_str(), message.size());
    }

    TEST(LogSinkTest, Composition)
    {
        CompositeLogSink composite{};
        auto sink1 = std::make_shared<TestLogSink>();
        auto sink2 = std::make_shared<TestLogSink>();

        // write to sink
        composite.AddLogSink(sink1);
        SendLogMessage(composite, LOTUS_LOG_SEVERITY_INFO, "hello");
        ASSERT_EQ(sink1->messages_.size(), 1);
        EXPECT_EQ(sink1->messages_[0], "hello");
        EXPECT_TRUE(sink2->messages_.empty());

        // write to two sinks
        composite.AddLogSink(sink2);
        SendLogMessage(composite, LOTUS_LOG_SEVERITY_INFO, "hi");
        ASSERT_EQ(sink1->messages_.size(), 2);
        EXPECT_EQ(sink1->messages_[1], "hi");
        ASSERT_EQ(sink2->messages_.size(), 1);
        EXPECT_EQ(sink2->messages_[0], "hi");
    }

    TEST(LogSinkTest, Default)
    {
        // this just exercises the default log sink, no real verification for now
        auto defaultLogSink = GetDefaultLogSink();
        ASSERT_NE(defaultLogSink, nullptr);

        const LogSeverity min_severity = -1, max_severity = LOTUS_LOG_SEVERITY_FATAL;
        for (int i = 0; i < 10; ++i)
        {
            const LogSeverity severity =
                min_severity + (i % (max_severity - min_severity + 1));
            SendLogMessage(*defaultLogSink, severity,
                           "hello from severity " + std::to_string(severity));
        }
    }
}
}