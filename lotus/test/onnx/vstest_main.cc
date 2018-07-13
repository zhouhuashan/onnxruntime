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

namespace Microsoft {
namespace VisualStudio {
namespace CppUnitTestFramework {
template <>
std::wstring ToString<>(const EXECUTE_RESULT& q) {
  switch (q) {
    case EXECUTE_RESULT::SUCCESS:
      return L"SUCCESS";
    case EXECUTE_RESULT::UNKNOWN_ERROR:
      return L"UNKNOWN_ERROR";
    case EXECUTE_RESULT::WITH_EXCEPTION:
      return L"WITH_EXCEPTION";
    case EXECUTE_RESULT::RESULT_DIFFERS:
      return L"RESULT_DIFFERS";
    case EXECUTE_RESULT::SHAPE_MISMATCH:
      return L"SHAPE_MISMATCH";
    case EXECUTE_RESULT::TYPE_MISMATCH:
      return L"TYPE_MISMATCH";
    case EXECUTE_RESULT::NOT_SUPPORT:
      return L"NOT_SUPPORT";
    case EXECUTE_RESULT::LOAD_MODEL_FAILED:
      return L"LOAD_MODEL_FAILED";
    case EXECUTE_RESULT::INVALID_GRAPH:
      return L"INVALID_GRAPH";
    case EXECUTE_RESULT::INVALID_ARGUMENT:
      return L"INVALID_ARGUMENT";
    case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
      return L"MODEL_SHAPE_MISMATCH";
    case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
      return L"MODEL_TYPE_MISMATCH";
  }
  return L"UNKNOWN_RETURN_CODE";
}
}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft
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

static void run(SessionFactory& sf) {
  char buf[1024];
  std::vector<EXECUTE_RESULT> res;
  {
    size_t requiredSize;
    getenv_s(&requiredSize, NULL, 0, "CloudTestJobName");
    Assert::AreNotEqual((size_t)0, requiredSize);
    std::string cloudTestJobName(requiredSize, '\0');
    getenv_s(&requiredSize, (char*)cloudTestJobName.data(), requiredSize, "CloudTestJobName");
    int p_models = Lotus::Env::Default().GetNumCpuCores();
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
    Lotus::AllocatorPtr cpu_allocator(new Lotus::CPUAllocator());
    std::vector<ITestCase*> tests = LoadTests({"..\\models"}, {modelName}, cpu_allocator);
    Assert::AreEqual((size_t)1, tests.size());
    HANDLE finish_event = CreateEvent(
        NULL,                // default security attributes
        TRUE,                // manual-reset event
        FALSE,               // initial state is nonsignaled
        TEXT("FinishEvent")  // object name
    );
    Assert::IsNotNull(finish_event);
    RunSingleTestCase(tests[0], sf, p_models, nullptr, [finish_event, &res](std::shared_ptr<TestCaseResult> result, PTP_CALLBACK_INSTANCE pci) {
      res = result->GetExcutionResult();
      return SetWindowsEvent(pci, finish_event);
    });
    DWORD dwWaitResult = WaitForSingleObject(finish_event, INFINITE);
    Assert::AreEqual(WAIT_OBJECT_0, dwWaitResult);
    CloseHandle(finish_event);
    Assert::AreEqual(tests[0]->GetDataCount(), res.size());
    delete tests[0];
  }
  for (EXECUTE_RESULT r : res) {
    Assert::AreEqual(EXECUTE_RESULT::SUCCESS, r);
  }
}

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
