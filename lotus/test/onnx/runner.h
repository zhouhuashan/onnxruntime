#pragma once
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/ml_value.h"
#include "core/platform/env_time.h"

#include "TestCase.h"
#include "TestCaseResult.h"
#include "testenv.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#ifdef _WIN32
typedef PTP_CALLBACK_INSTANCE LOTUS_CALLBACK_INSTANCE;
#else
typedef void* LOTUS_CALLBACK_INSTANCE;
#endif

typedef std::function<Lotus::Common::Status(std::shared_ptr<TestCaseResult> result, LOTUS_CALLBACK_INSTANCE pci)> TestCaseCallBack;

class DataRunner {
 protected:
  typedef TestCaseCallBack CALL_BACK;
  std::shared_ptr<TestCaseResult> result;
  std::string test_case_name_;
  ITestCase* c_;
  //Time spent in Session::Run. It only make sense when SeqTestRunner was used
  Lotus::TIME_SPEC spent_time_;

 private:
  std::shared_ptr<Lotus::InferenceSession> session;
  CALL_BACK on_finished;
  EXECUTE_RESULT RunTaskImpl(size_t task_id);

 public:
  DataRunner(std::shared_ptr<Lotus::InferenceSession> session1, const std::string& test_case_name1, ITestCase* c, TestCaseCallBack on_finished1);
  virtual void OnTaskFinished(size_t task_id, EXECUTE_RESULT res, LOTUS_CALLBACK_INSTANCE pci) noexcept = 0;
  void RunTask(size_t task_id, LOTUS_CALLBACK_INSTANCE pci, bool store_result);
  virtual ~DataRunner() {}

  virtual void Start(size_t concurrent_runs) = 0;

  void finish(std::shared_ptr<TestCaseResult> res, LOTUS_CALLBACK_INSTANCE pci) {
    CALL_BACK callback = on_finished;
    res->SetSpentTime(spent_time_);
    for (EXECUTE_RESULT r : result->GetExcutionResult()) {
      switch (r) {
        case EXECUTE_RESULT::RESULT_DIFFERS:
          LOGF_DEFAULT(ERROR, "%s: result differs\n", test_case_name_.c_str());
          break;
        case EXECUTE_RESULT::SHAPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: shape mismatch\n", test_case_name_.c_str());
          break;
        case EXECUTE_RESULT::TYPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: type mismatch\n", test_case_name_.c_str());
          break;
        case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: shape in model file mismatch\n", test_case_name_.c_str());
          break;
        case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
          LOGF_DEFAULT(ERROR, "%s: type in model file mismatch\n", test_case_name_.c_str());
        default:
          //nothing to do
          break;
      }
      if (r != EXECUTE_RESULT::SUCCESS) break;
    }
    delete this;
    callback(res, pci);
  }
};

class SeqTestRunner : public DataRunner {
 private:
  size_t repeat_count_;

 public:
  SeqTestRunner(std::shared_ptr<Lotus::InferenceSession> session1,
                ITestCase* c, size_t repeat_count,
                TestCaseCallBack on_finished1);

  void Start(size_t concurrent_runs) override;
  void OnTaskFinished(size_t, EXECUTE_RESULT, LOTUS_CALLBACK_INSTANCE) noexcept override {}
};

class PTestRunner : public DataRunner {
 private:
  std::atomic<size_t> next_test_to_run;
  std::atomic<size_t> finished;
  void OnTaskFinished(size_t task_id, EXECUTE_RESULT res, LOTUS_CALLBACK_INSTANCE pci) noexcept override;

 public:
  void Start(size_t concurrent_runs) override;

  PTestRunner(std::shared_ptr<Lotus::InferenceSession> session1,
              ITestCase* c,
              TestCaseCallBack on_finished1);

 private:
  bool ScheduleNew();
};

std::vector<ITestCase*> LoadTests(const std::vector<std::string>& input_paths, const std::vector<std::string>& whitelisted_test_cases, Lotus::AllocatorPtr allocator);
Lotus::Common::Status RunTests(TestEnv& env, int p_models, int concurrent_runs, size_t repeat_count);

EXECUTE_RESULT StatusCodeToExecuteResult(int input);

#ifdef _WIN32
extern void ParallelRunTests(TestEnv& env, int p_models, size_t concurrent_runs, size_t repeat_count);
Lotus::Common::Status SetWindowsEvent(LOTUS_CALLBACK_INSTANCE pci, HANDLE finish_event);
#endif
void RunSingleTestCase(ITestCase* info, const SessionFactory& sf, size_t concurrent_runs, size_t repeat_count, LOTUS_CALLBACK_INSTANCE pci, TestCaseCallBack on_finished);
