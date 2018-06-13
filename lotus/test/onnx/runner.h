#pragma once
#include <vector>
#include <string>
#include "testenv.h"
#include "core/framework/ml_value.h"
#include "core/common/common.h"
#include "TestCase.h"

class DataRunner {
 protected:
  typedef std::function<void(std::shared_ptr<TestCaseResult>)> CALL_BACK;
  std::shared_ptr<TestCaseResult> result;
  std::string test_case_name_;
  ITestCase* c_;
  Lotus::TIME_SPEC spent_time_;

 private:
  std::shared_ptr<Lotus::InferenceSession> session;
  CALL_BACK on_finished;
  void RunTaskImpl(size_t task_id);
  void SetResult(size_t task_id, EXECUTE_RESULT result) noexcept;

 public:
  DataRunner(std::shared_ptr<Lotus::InferenceSession> session1, const std::string& test_case_name1, ITestCase* c, std::function<void(std::shared_ptr<TestCaseResult> result)> on_finished1);
  virtual void OnTaskFinished(size_t task_id, EXECUTE_RESULT res) noexcept = 0;
  virtual ~DataRunner() {}
  void RunTask(size_t task_id);
  virtual void Start(size_t concurrent_runs) = 0;

  void finish(std::shared_ptr<TestCaseResult> res) {
    CALL_BACK callback = on_finished;
    res->SetSpentTime(spent_time_);
    delete this;
    callback(res);
  }
};

class SeqTestRunner : public DataRunner {
 public:
  SeqTestRunner(std::shared_ptr<Lotus::InferenceSession> session1,
                ITestCase* c,
                std::function<void(std::shared_ptr<TestCaseResult> result)> on_finished1);

  void Start(size_t concurrent_runs) override;
  void OnTaskFinished(size_t, EXECUTE_RESULT) noexcept override {}
};

class PTestRunner : public DataRunner {
 private:
  std::atomic<size_t> next_test_to_run;
  std::atomic<size_t> finished;
  void OnTaskFinished(size_t task_id, EXECUTE_RESULT res) noexcept override;

 public:
  void Start(size_t concurrent_runs) override;

  PTestRunner(std::shared_ptr<Lotus::InferenceSession> session1,
              ITestCase* c,
              std::function<void(std::shared_ptr<TestCaseResult> result)> on_finished1);

 private:
  bool ScheduleNew();
};

void RunSingleTestCase(TestEnv& env, size_t test_index, size_t concurrent_runs, std::function<void(std::shared_ptr<TestCaseResult>)> on_finished);
std::vector<ITestCase*> LoadTests(const std::vector<std::string>& input_paths, const std::vector<std::string>& whitelisted_test_cases, Lotus::AllocatorPtr allocator);
void RunTests(TestEnv& env, int p_models, int concurrent_runs);

EXECUTE_RESULT StatusCodeToExecuteResult(int input);

#ifdef _WIN32
extern void ParallelRunTests(TestEnv& env, int p_models, size_t concurrent_runs);
#endif
