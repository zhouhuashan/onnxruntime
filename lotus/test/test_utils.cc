#include "test/test_utils.h"

#include <iostream>
#include <memory>

#include "gtest/gtest.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

using namespace Lotus::Logging;

namespace Lotus {
namespace Test {

Lotus::Logging::LoggingManager& DefaultLoggingManager() {
  // create a CLog based default logging manager
  static std::string default_logger_id{"Default"};
  static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CLogSink{}},
                                                Severity::kVERBOSE, false, LoggingManager::InstanceType::Default,
                                                &default_logger_id};
  return default_logging_manager;
}

void DefaultInitialize(int argc, char** argv, bool create_default_logging_manager) {
  std::clog << "Initializing unit testing." << std::endl;
  testing::InitGoogleTest(&argc, argv);

  if (create_default_logging_manager) {
    // make sure default logging manager exists and is working
    auto logger = Lotus::Test::DefaultLoggingManager().DefaultLogger();

    LOGS(logger, VERBOSE) << "Logging manager initialized.";
  }
}

}  // namespace Test
}  // namespace Lotus
