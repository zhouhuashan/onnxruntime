#pragma once
#include <string>
#include <vector>
#include <experimental/filesystem>  // C++-standard header file name
#ifdef _MSC_VER
#include <filesystem>
#endif

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/ml_value.h"
#include "core/platform/env_time.h"

#include "TestCase.h"
#include "TestCaseResult.h"

#include "testenv.h"
#include <core/graph/graph.h>  //TODO(@chasun): remove this
#include "sync_api.h"

typedef std::function<::onnxruntime::common::Status(std::shared_ptr<TestCaseResult> result, LOTUS_CALLBACK_INSTANCE pci)> TestCaseCallBack;

struct TestCaseTask {
  TestEnv& env;
  const int task_id;
  //The max number of concurrent Session::Run() for each model
  const size_t concurrent_runs;
  const size_t repeat_count;
  const PThreadPool pool;
};

void LOTUS_CALLBACK RunTestCase(LOTUS_CALLBACK_INSTANCE instance, void* context, LOTUS_WORK work);
//TODO: implement this function for Linux
void LOTUS_CALLBACK RunSingleDataItem(LOTUS_CALLBACK_INSTANCE instance, void* context, LOTUS_WORK work);
::onnxruntime::common::Status OnTestCaseFinished(LOTUS_CALLBACK_INSTANCE pci, TestCaseTask* task, std::shared_ptr<TestCaseResult> result);

class DataRunner {
 protected:
  typedef TestCaseCallBack CALL_BACK;
  std::shared_ptr<TestCaseResult> result;
  std::string test_case_name_;
  ITestCase* c_;
  //Time spent in Session::Run. It only make sense when SeqTestRunner was used
  ::onnxruntime::TIME_SPEC spent_time_;

 private:
  std::shared_ptr<::onnxruntime::InferenceSession> session;
  CALL_BACK on_finished;
  EXECUTE_RESULT RunTaskImpl(size_t task_id);

 public:
  DataRunner(std::shared_ptr<::onnxruntime::InferenceSession> session1, const std::string& test_case_name1, ITestCase* c, TestCaseCallBack on_finished1);
  virtual void OnTaskFinished(size_t task_id, EXECUTE_RESULT res, LOTUS_CALLBACK_INSTANCE pci) noexcept = 0;
  void RunTask(size_t task_id, LOTUS_CALLBACK_INSTANCE pci, bool store_result);
  virtual ~DataRunner() {}

  virtual void Start(LOTUS_CALLBACK_INSTANCE pci, size_t concurrent_runs) = 0;

  void finish(LOTUS_CALLBACK_INSTANCE pci) {
    std::shared_ptr<TestCaseResult> res = result;
    CALL_BACK callback = on_finished;
    res->SetSpentTime(spent_time_);
    const std::vector<EXECUTE_RESULT>& er = res->GetExcutionResult();
    for (size_t i = 0; i != er.size(); ++i) {
      EXECUTE_RESULT r = er[i];
      if (r == EXECUTE_RESULT::SUCCESS) continue;
      std::string s = c_->GetDatasetDebugInfoString(i);
      switch (r) {
        case EXECUTE_RESULT::RESULT_DIFFERS:
          LOGF_DEFAULT(ERROR, "%s: result differs. Dataset:%s\n", test_case_name_.c_str(), s.c_str());
          break;
        case EXECUTE_RESULT::SHAPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: shape mismatch. Dataset:%s\n", test_case_name_.c_str(), s.c_str());
          break;
        case EXECUTE_RESULT::TYPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: type mismatch. Dataset:%s\n", test_case_name_.c_str(), s.c_str());
          break;
        case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: shape in model file mismatch. Dataset:%s\n", test_case_name_.c_str(), s.c_str());
          break;
        case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: type in model file mismatch. Dataset:%s\n", test_case_name_.c_str(), s.c_str());
        default:
          //nothing to do
          break;
      }
      break;
    }
    delete this;
    callback(res, pci);
  }
};

class SeqTestRunner : public DataRunner {
 private:
  size_t repeat_count_;

 public:
  SeqTestRunner(std::shared_ptr<::onnxruntime::InferenceSession> session1,
                ITestCase* c, size_t repeat_count,
                TestCaseCallBack on_finished1);

  void Start(LOTUS_CALLBACK_INSTANCE pci, size_t concurrent_runs) override;
  void OnTaskFinished(size_t, EXECUTE_RESULT, LOTUS_CALLBACK_INSTANCE) noexcept override {}
};

class PTestRunner : public DataRunner {
 private:
  std::atomic<size_t> next_test_to_run;
  std::atomic<size_t> finished;
  void OnTaskFinished(size_t task_id, EXECUTE_RESULT res, LOTUS_CALLBACK_INSTANCE pci) noexcept override;

 public:
  void Start(LOTUS_CALLBACK_INSTANCE pci, size_t concurrent_runs) override;

  PTestRunner(std::shared_ptr<::onnxruntime::InferenceSession> session1,
              ITestCase* c, PThreadPool tpool,
              TestCaseCallBack on_finished1);

 private:
  bool ScheduleNew();
  const PThreadPool tpool_;
};

struct DataTask {
  PTestRunner* env;
  const size_t task_id;
};

std::vector<ITestCase*> LoadTests(const std::vector<std::experimental::filesystem::v1::path>& input_paths, const std::vector<std::string>& whitelisted_test_cases, ::onnxruntime::AllocatorPtr allocator);
//Do not run this function in the thread pool passed in
::onnxruntime::common::Status RunTests(TestEnv& env, int p_models, int concurrent_runs, size_t repeat_count, PThreadPool tpool);
EXECUTE_RESULT StatusCodeToExecuteResult(int input);
void RunSingleTestCase(ITestCase* info, const SessionFactory& sf, size_t concurrent_runs, size_t repeat_count, PThreadPool tpool, LOTUS_CALLBACK_INSTANCE pci, TestCaseCallBack on_finished);
