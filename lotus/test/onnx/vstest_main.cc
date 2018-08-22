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
using std::experimental::filesystem::v1::path;

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

static void run(SessionFactory& sf, const wchar_t* test_folder) {
  char buf[1024];
  std::vector<EXECUTE_RESULT> res;
  {
    //Current working directory is the one who contains 'onnx_test_runner_vstest.dll'
    //We want debug build and release build share the same test data files, so it should
    //be one level up.
    ::Lotus::AllocatorPtr cpu_allocator(new ::Lotus::CPUAllocator());
    std::wstring test_folder_full_path(L"..\\models\\");
    test_folder_full_path.append(test_folder);
    path p1(test_folder_full_path);
    std::vector<ITestCase*> tests = LoadTests({p1}, {}, cpu_allocator);
    Assert::AreEqual((size_t)1, tests.size());
    int p_models = ::Lotus::Env::Default().GetNumCpuCores();
    if (tests[0]->GetTestCaseName() == "coreml_FNS-Candy_ImageNet") {
      p_models = 2;
    }
    snprintf(buf, sizeof(buf), "running test %s with %d cores", tests[0]->GetTestCaseName().c_str(), p_models);
    Logger::WriteMessage(buf);
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
#include "vsts/tests.inc"
};
