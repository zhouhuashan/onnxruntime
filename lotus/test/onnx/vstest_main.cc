#include <CppUnitTest.h>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <core/platform/env.h>
#include <core/graph/onnx_protobuf.h>
#include "runner.h"

#include <core/framework/environment.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/graph/constants.h>

#include "vstest_logger.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

static std::unique_ptr<::Lotus::Environment> env;
::Lotus::Logging::LoggingManager* default_logger;
static std::string logger_id("onnx_test_runner");

TEST_MODULE_INITIALIZE(ModuleInitialize) {
  Logger::WriteMessage("Initialize Lotus");
  default_logger = new ::Lotus::Logging::LoggingManager{std::unique_ptr<::Lotus::Logging::ISink>{new VsTestSink{}},
                                                      ::Lotus::Logging::Severity::kWARNING, false,
                                                      ::Lotus::Logging::LoggingManager::InstanceType::Default,
                                                      &logger_id};
  auto status = ::Lotus::Environment::Create(env);
  if (!status.IsOK()) {
    Logger::WriteMessage(status.ErrorMessage().c_str());
    Logger::WriteMessage("Create ::Lotus::Environment failed");
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

static void run(SessionFactory& sf) {
  char buf[1024];
  std::vector<EXECUTE_RESULT> res;
  {
    size_t requiredSize;
    getenv_s(&requiredSize, NULL, 0, "CloudTestJobName");
    Assert::AreNotEqual((size_t)0, requiredSize);
    std::string cloudTestJobName(requiredSize, '\0');
    getenv_s(&requiredSize, (char*)cloudTestJobName.data(), requiredSize, "CloudTestJobName");
    int p_models = ::Lotus::Env::Default().GetNumCpuCores();
    snprintf(buf, sizeof(buf), "running test %s with %d cores", cloudTestJobName.c_str(), p_models);
    Logger::WriteMessage(buf);
    size_t pos1 = cloudTestJobName.find('.', 0);
    Assert::AreNotEqual(std::string::npos, requiredSize);
    ++pos1;
    size_t pos2 = cloudTestJobName.find('.', pos1);
    Assert::AreNotEqual(std::string::npos, requiredSize);
    std::string modelName = cloudTestJobName.substr(pos1, pos2 - pos1);
    snprintf(buf, sizeof(buf), "model %s", modelName.c_str());
    Logger::WriteMessage(buf);
    //CloudTestJobName
    //Current working directory is the one who contains 'onnx_test_runner_vstest.dll'
    //We want debug build and release build share the same test data files, so it should
    //be one level up.
    ::Lotus::AllocatorPtr cpu_allocator(new ::Lotus::CPUAllocator());
    std::vector<ITestCase*> tests = LoadTests({"..\\models"}, {modelName}, cpu_allocator);
    Assert::AreEqual((size_t)1, tests.size());
    LOTUS_EVENT finish_event;
    ::Lotus::Status status = CreateLotusEvent(&finish_event);
    Assert::IsTrue(status.IsOK());
    Assert::IsNotNull(finish_event);
    RunSingleTestCase(tests[0], sf, p_models, 1, GetDefaultThreadPool(::Lotus::Env::Default()), nullptr, [finish_event, &res](std::shared_ptr<TestCaseResult> result, PTP_CALLBACK_INSTANCE pci) {
      res = result->GetExcutionResult();
      return LotusSetEventWhenCallbackReturns(pci, finish_event);
    });
    status = WaitAndCloseEvent(finish_event);
    Assert::IsTrue(status.IsOK());
    Assert::AreEqual(tests[0]->GetDataCount(), res.size());
    delete tests[0];
  }
  for (EXECUTE_RESULT r : res) {
    Assert::AreEqual(EXECUTE_RESULT::SUCCESS, r);
  }
}
// clang-format off
TEST_CLASS(ONNX_TEST){
  public :
    TEST_METHOD(normal_run){
      SessionFactory sf(LotusIR::kCpuExecutionProvider, true, true);
      run(sf);
    }

    TEST_METHOD(disable_cpu_mem_arena) {
      SessionFactory sf(LotusIR::kCpuExecutionProvider, true, false);
      run(sf);
    }
};
