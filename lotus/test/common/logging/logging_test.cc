#include <exception>
#include <functional>
#include <string>

#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

#include "test/common/logging/helpers.h"

using namespace Lotus;
using namespace Lotus::Logging;
using InstanceType = LoggingManager::InstanceType;

// if we pull in the whole 'testing' namespace we get warnings from date.h as both use '_' in places.
// to avoid that we explicitly pull in the pieces we are using
using testing::Eq;
using testing::Field;
using testing::Ge;
using testing::HasSubstr;
using testing::Property;

static std::string default_logger_id{"TestFixtureDefaultLogger"};

// class to provide single default instance of LoggingManager for use with macros involving 'DEFAULT'
class LoggingTestsFixture : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    // logger uses kWARNING so we can test filtering of kVERBOSE output,
    // and filters user data so that can also be tested
    const bool filter_user_data = true;

    default_logging_manager_ = std::make_unique<LoggingManager>(
        std::unique_ptr<ISink>{new CLogSink{}}, Severity::kWARNING, filter_user_data,
        InstanceType::Default, &default_logger_id);
  }

  static void TearDownTestCase() {
    default_logging_manager_.release();
  }

  // Objects declared here can be used by all tests in the test case for Foo.
  static unique_ptr<LoggingManager> default_logging_manager_;
};

unique_ptr<LoggingManager> LoggingTestsFixture::default_logging_manager_;

/// <summary>
/// Tests that the WHERE macro populates all fields correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestWhereMacro) {
  const std::string logid{"TestWhereMacro"};
  const std::string message{"Testing the WHERE macro."};
  const Severity min_log_level = Severity::kVERBOSE;
  const Severity log_level = Severity::kERROR;

  const std::string file = __FILE__;
  const std::string function = __PRETTY_FUNCTION__;
  int log_line = 0;

  std::cout << function << std::endl;

  MockSink *sinkPtr = new MockSink();

  EXPECT_CALL(*sinkPtr, SendImpl(testing::_, HasSubstr(logid),
                                 Property(&Capture::Location,
                                          AllOf(Field(&CodeLocation::line_num, Eq(std::ref(log_line))),
                                                Field(&CodeLocation::file_and_path, HasSubstr("lotus")),            // path
                                                Field(&CodeLocation::file_and_path, HasSubstr("logging_test.cc")),  // filename
                                                Field(&CodeLocation::function, HasSubstr("TestWhereMacro"))))))
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sinkPtr), min_log_level, false, InstanceType::Temporal};

  std::unique_ptr<Logger> logger = manager.CreateLogger(logid);

  log_line = __LINE__ + 1;
  LOGS(*logger, ERROR) << message;
}

/// <summary>
/// Tests that the logging manager filters based on severity and user data correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestFiltering) {
  const std::string logid{"TestFiltering"};
  const Severity minLogLevel = Severity::kWARNING;
  const bool filter_user_data = true;

  MockSink *sinkPtr = new MockSink();

  EXPECT_CALL(*sinkPtr, SendImpl(testing::_, HasSubstr(logid), testing::_))  // Property(&Capture::Severity, Ge(minLogLevel))))
      .Times(1)
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sinkPtr), minLogLevel, filter_user_data,
                         InstanceType::Temporal};

  auto logger = manager.CreateLogger(logid);

  LOGS(*logger, VERBOSE) << "Filtered by severity";
  LOGS_USER(*logger, ERROR) << "Filtered user data";
  LOGF(*logger, ERROR, "%s", "hello");  // not filtered
  LOGF_USER(*logger, ERROR, "Filtered %s", "user data");

  LOGS_DEFAULT(WARNING) << "Warning";  // not filtered
  LOGS_USER_DEFAULT(ERROR) << "Filtered user data";
  LOGF_DEFAULT(VERBOSE, "Filtered by severity");
  LOGF_USER_DEFAULT(WARNING, "Filtered user data");
}

/// <summary>
/// Tests that the logging manager constructor validates its usage correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestLoggingManagerCtor) {
  // throw if sink is null
  EXPECT_THROW((LoggingManager{std::unique_ptr<ISink>{nullptr}, Severity::kINFO, false,
                               InstanceType::Temporal}),
               std::logic_error);

  // can't have two logging managers with InstanceType of Default.
  // this should clash with LoggingTestsFixture::default_logging_manager_
  EXPECT_THROW((LoggingManager{std::unique_ptr<ISink>{new MockSink{}}, Severity::kINFO, false,
                               InstanceType::Default}),
               std::logic_error);
}

/// <summary>
/// Tests that the conditional logging macros work correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestConditionalMacros) {
  const std::string logger_id{"TestConditionalMacros"};
  const Severity minLogLevel = Severity::kVERBOSE;
  const bool filter_user_data = false;

  MockSink *sinkPtr = new MockSink();

  // two logging calls that are true using default logger which won't hit our MockSink

  // two logging calls that are true using non-default logger
  EXPECT_CALL(*sinkPtr, SendImpl(testing::_, HasSubstr(logger_id), testing::_))
      .Times(2)
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sinkPtr), minLogLevel, filter_user_data,
                         InstanceType::Temporal};

  auto logger = manager.CreateLogger(logger_id);

  // macros to use local logger
  LOGS_IF(logger == nullptr, *logger, INFO) << "Null logger";                    // false
  LOGS_IF(logger != nullptr, *logger, INFO) << "Valid logger";                   // true
  LOGF_IF(logger == nullptr, *logger, INFO, "Logger is %p", logger.get());       // false
  LOGF_USER_IF(logger != nullptr, *logger, INFO, "Logger is %p", logger.get());  // true

  // macros to test LoggingTestsFixture::default_logging_manager_
  LOGS_DEFAULT_IF(logger == nullptr, INFO) << "Null logger";                    // false
  LOGS_USER_DEFAULT_IF(logger != nullptr, INFO) << "Valid logger";              // true but user data filtered
  LOGF_DEFAULT_IF(logger == nullptr, INFO, "Logger is %p", logger.get());       // false
  LOGF_USER_DEFAULT_IF(logger != nullptr, INFO, "Logger is %p", logger.get());  // true but user data filtered
}

/// <summary>
/// Tests that the VLOG* macros produce the expected output.
/// Disabled in Release build, so should be no calls to SendImpl in that case.
/// </summary>
TEST_F(LoggingTestsFixture, TestVLog) {
  const std::string logid{"TestVLog"};

  MockSink *sinkPtr = new MockSink();

  // we only get the non-default calls from below in this sink
  EXPECT_CALL(*sinkPtr, SendImpl(testing::_, HasSubstr(logid), testing::_))
#ifdef _DEBUG
      .Times(2)
      .WillRepeatedly(PrintArgs());
#else
      .Times(0);
#endif

  LoggingManager manager{std::unique_ptr<ISink>(sinkPtr), Severity::kVERBOSE, false, InstanceType::Temporal};

  auto logger = manager.CreateLogger(logid);

  // test non-default
  VLOGS(*logger, 1) << "Stream";
  VLOGF(*logger, 2, "Printf %d", 1);

  VLOGS_USER(*logger, Logging::max_vlog_level) << "User data";    // ignored
  VLOGF_USER(*logger, Logging::max_vlog_level, "User Id %d", 1);  // ignored

  // test default
  VLOGS_DEFAULT(1) << "Stream";
  VLOGF_DEFAULT(2, "Printf %d", 1);

  VLOGS_USER_DEFAULT(Logging::max_vlog_level) << "User data";    // ignored
  VLOGF_USER_DEFAULT(Logging::max_vlog_level, "User Id %d", 1);  // ignored

#ifdef _DEBUG
  // test we can set the max level
  max_vlog_level = INT_MIN;
  VLOGS(*logger, 0) << "Should be ignored.";
#endif
}

/*
/// <summary>
/// Tests the *kFatal* macros.
/// </summary>
TEST_F(LoggingTestsFixture, TestFatalMacros) {
  const std::string logid{"TestFatalMacros"};

  MockSink *sinkPtr = new MockSink();

  EXPECT_CALL(*sinkPtr, SendImpl(testing::_, testing::_, testing::_))
      .Times(2)
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sinkPtr), Severity::kVERBOSE, false, logid};

  EXPECT_THROW(FATAL("Category:Test", "Severity::Fatal should throw"), std::runtime_error);
  EXPECT_THROW(FATAL_IF(logid.length() == 15), std::runtime_error);
}

TEST_F(LoggingTestsFixture, TestNotNull) {
  const std::string logid{"NotNull"};

  MockSink *sinkPtr = new MockSink();

  EXPECT_CALL(*sinkPtr, SendImpl(testing::_, testing::_, testing::_))
      .Times(5)  // 5 failed checks, 2 succeed so no logging
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sinkPtr), Severity::kVERBOSE, false, logid};

  // test smart and raw pointers
  std::shared_ptr<std::string> strptr1;
  std::unique_ptr<std::string> strptr2;
  std::string *strptr3 = nullptr;

  EXPECT_THROW(CHECK_NOTNULL(strptr1), std::runtime_error);
  EXPECT_THROW(CHECK_NOTNULL(strptr2), std::runtime_error);
  EXPECT_THROW(CHECK_NOTNULL(strptr3), std::runtime_error);

  // validate some CHECK_OP based tests
  EXPECT_THROW(CHECK_EQ(1, 2), std::runtime_error);
  EXPECT_THROW(CHECK_GE(1, 2), std::runtime_error);
  EXPECT_NO_THROW(CHECK_NE(1, 2));
  EXPECT_NO_THROW(CHECK_LT(1, 2));
}
*/
