// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/test_environment.h"

#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

using namespace ::onnxruntime::Logging;

namespace onnxruntime {
namespace Test {

static LoggingManager* s_default_logging_manager = nullptr;

::onnxruntime::Logging::LoggingManager& DefaultLoggingManager() {
  LOTUS_ENFORCE(s_default_logging_manager != nullptr,
                "Need a TestEnvironment instance to provide the default logging manager.");

  return *s_default_logging_manager;
}

TestEnvironment::TestEnvironment(int argc, char** argv, bool create_default_logging_manager) {
  LOTUS_ENFORCE(s_default_logging_manager == nullptr,
                "Only expected one instance of TestEnvironment to be created.");

  std::clog << "Initializing unit testing." << std::endl;
  testing::InitGoogleTest(&argc, argv);

  if (create_default_logging_manager) {
    static std::string default_logger_id{"Default"};
    logging_manager_ = std::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                        Severity::kWARNING,  // TODO make this configurable through
                                                                             // cmd line arguments or some other way
                                                        false,
                                                        LoggingManager::InstanceType::Default,
                                                        &default_logger_id);

    s_default_logging_manager = logging_manager_.get();

    // make sure default logging manager exists and is working
    auto logger = ::onnxruntime::Test::DefaultLoggingManager().DefaultLogger();
    LOGS(logger, VERBOSE) << "Logging manager initialized.";
  }

#ifdef HAVE_FRAMEWORK_LIB
  auto status = Environment::Create(runtime_environment_);
  LOTUS_ENFORCE(status == Status::OK(), "Failed creating runtime environment. ", status.ErrorMessage());
#endif
}

TestEnvironment::~TestEnvironment() {
#ifdef HAVE_FRAMEWORK_LIB
  // release environment followed by logging manager so any log output from runtime environment shutdown
  // using the default logger will succeed.
  runtime_environment_ = nullptr;
#else
  ::google::protobuf::ShutdownProtobufLibrary();
#endif

  s_default_logging_manager = nullptr;
  logging_manager_ = nullptr;
}

}  // namespace Test
}  // namespace onnxruntime
