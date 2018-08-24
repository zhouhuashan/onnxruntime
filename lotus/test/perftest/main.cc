// Lotus dependencies
#include <core/common/logging/sinks/clog_sink.h>
#include <core/common/logging/logging.h>
#include <core/framework/environment.h>
#include <core/platform/env.h>

#include "command_args_parser.h"
#include "performance_runner.h"

using namespace LotusIR;
using namespace Lotus;

int main(int argc, char* args[]) {
  std::string default_logger_id{"Default"};
  Logging::LoggingManager default_logging_manager{std::unique_ptr<Logging::ISink>{new Logging::CLogSink{}},
                                                  Logging::Severity::kWARNING, false,
                                                  Logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  std::unique_ptr<Environment> env;
  auto status = Environment::Create(env);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "failed to create environment:%s", status.ErrorMessage().c_str());
    return -1;
  }

  ::Lotus::PerfTest::PerformanceTestConfig test_config;
  if (!::Lotus::PerfTest::CommandLineParser::ParseArguments(test_config, argc, args)) {
    ::Lotus::PerfTest::CommandLineParser::ShowUsage();
    return -1;
  }

  ::Lotus::PerfTest::PerformanceRunner perf_runner(test_config);
  status = perf_runner.Run();
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "Run failed:%s", status.ErrorMessage().c_str());
    return -1;
  }

  perf_runner.SerializeResult();

  return 0;
}
