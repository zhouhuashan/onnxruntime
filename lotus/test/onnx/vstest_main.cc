#include <CppUnitTest.h>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <core/platform/env.h>
#include "runner.h"
#include <core/framework/environment.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/graph/constants.h>
#include "vstest_logger.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

static std::unique_ptr<Lotus::Environment> env;
Lotus::Logging::LoggingManager* default_logger;
static std::string logger_id(__FILE__);

TEST_MODULE_INITIALIZE(ModuleInitialize) {
  Logger::WriteMessage("Initialize Lotus");
  default_logger = new Lotus::Logging::LoggingManager{std::unique_ptr<Lotus::Logging::ISink>{new VsTestSink{}},
                                                      Lotus::Logging::Severity::kWARNING, false,
                                                      Lotus::Logging::LoggingManager::InstanceType::Default,
                                                      &logger_id};
  auto status = Lotus::Environment::Create(env);
  if (!status.IsOK()) {
    Logger::WriteMessage(status.ErrorMessage().c_str());
    Logger::WriteMessage("Create Lotus::Environment failed");
    abort();
  }
}

TEST_MODULE_CLEANUP(ModuleCleanup) {
  Logger::WriteMessage("Cleanup Lotus");
  env.reset();
  delete default_logger;
}

TEST_CLASS(ONNX_TEST){
    template <Lotus::AllocationPlannerType planner>
    void run(const std::string& provider){
        char buf[1024];
int p_models = Lotus::Env::Default()->GetNumCpuCores();
snprintf(buf, sizeof(buf), "running tests with %d cores", p_models);
Logger::WriteMessage(buf);
std::vector<TestCaseInfo> tests = LoadTests({"onnx_testdata"}, {});
TestResultStat stat;
TestEnv args(tests, stat, planner, provider);
RunTests(args, p_models, p_models);
std::string res = stat.ToString();
Logger::WriteMessage(res.c_str());
}

public:
TEST_METHOD(test_sequential_planner) {
  run<Lotus::AllocationPlannerType::SEQUENTIAL_PLANNER>(LotusIR::kCpuExecutionProvider);
}
TEST_METHOD(test_simple_sequential_planner) {
  run<Lotus::AllocationPlannerType::SIMPLE_SEQUENTIAL_PLANNER>(LotusIR::kCpuExecutionProvider);
}
}
;
