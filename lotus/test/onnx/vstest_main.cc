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
static std::string logger_id("onnx_test_runner");

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
  Logger::WriteMessage("Initialize Lotus finished");
}

TEST_MODULE_CLEANUP(ModuleCleanup) {
  Logger::WriteMessage("Cleanup Lotus");
  env.reset();
  delete default_logger;
  Logger::WriteMessage("Cleanup Lotus finished");
}

static void run(const std::string& provider) {
  char buf[1024];
  int p_models = Lotus::Env::Default().GetNumCpuCores();
  snprintf(buf, sizeof(buf), "running tests with %d cores", p_models);
  Logger::WriteMessage(buf);
  //Current working directory is the one who contains 'onnx_test_runner_vstest.dll'
  //We want debug build and release build share the same test data files, so it should
  //be one level up.
  Lotus::AllocatorPtr cpu_allocator(new Lotus::CPUAllocator());
  std::vector<ITestCase*> tests = LoadTests({ "..\\models" }, {}, cpu_allocator);  
  TestResultStat stat;
  SessionFactory sf(provider);
  TestEnv args(tests, stat, sf);
  RunTests(args, p_models, p_models);
  std::string res = stat.ToString();
  Logger::WriteMessage(res.c_str());
  for (ITestCase* l : tests) {
	  delete l;
  }
  size_t failed = stat.total_test_case_count - stat.succeeded - stat.skipped - stat.not_implemented;
  if (failed != 0) {
    Assert::Fail(L"test failed");
  }
}

TEST_CLASS(ONNX_TEST){
  public :
      TEST_METHOD(test_sequential_planner){
          run(LotusIR::kCpuExecutionProvider);
      }
};
