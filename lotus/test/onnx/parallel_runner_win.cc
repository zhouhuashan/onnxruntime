#include "testenv.h"
#include "runner.h"
#include "core/common/common.h"
#include <Windows.h>
#include "FixedCountFinishCallbackWin.h"
namespace {
struct TestCaseTask {
  TestEnv& env;
  const int task_id;
  //The max number of concurrent Session::Run() for each model
  const size_t concurrent_runs;
};

struct DataTask {
  PTestRunner* env;
  const size_t task_id;
};

void __stdcall RunTestCase(
    _Inout_ PTP_CALLBACK_INSTANCE,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work);

Lotus::Common::Status OnTestCaseFinished(PTP_CALLBACK_INSTANCE pci, TestCaseTask* task, std::shared_ptr<TestCaseResult> result) {
  FixedCountFinishCallback* finished = task->env.finished;
  auto task_id = task->task_id;
  bool failed = false;
  {
    std::unique_ptr<TestCaseTask> unused(task);
    TestEnv& env = task->env;
    int next_test = env.next_test_to_run++;
    if (next_test < env.tests.size()) {
      //schedule the next TestCase
      std::unique_ptr<TestCaseTask> t(new TestCaseTask{env, next_test, task->concurrent_runs});
      PTP_WORK work = CreateThreadpoolWork(RunTestCase, t.get(), nullptr);
      if (!work) {
        LOGF_DEFAULT(ERROR, "schedule test task failed\n");
        failed = true;
      } else {
        SubmitThreadpoolWork(work);
        t.release();  //ownership transferred to the ThreadpoolWork
      }
    }
  }
  if (failed)
    return finished->fail(pci);
  else
    return finished->onFinished(task_id, result, pci);
}

void __stdcall RunSingleDataItem(
    _Inout_ PTP_CALLBACK_INSTANCE pci,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work) {
  CloseThreadpoolWork(work);
  DataTask* task((DataTask*)context);
  PTestRunner* env = task->env;
  const size_t task_id = task->task_id;
  delete task;
  env->RunTask(task_id, pci);
}

void __stdcall RunTestCase(
    _Inout_ PTP_CALLBACK_INSTANCE pci,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work) {
  CloseThreadpoolWork(work);
  TestCaseTask* task((TestCaseTask*)context);
  ITestCase* info = task->env.tests[task->task_id];
  try {
    RunSingleTestCase(info, task->env.sf, task->concurrent_runs, pci, [task](std::shared_ptr<TestCaseResult> result, PTP_CALLBACK_INSTANCE pci) {
      return OnTestCaseFinished(pci, task, result);
    });
    return;
  } catch (std::exception& ex) {
    LOGF_DEFAULT(ERROR, ex.what());
  }
  std::shared_ptr<TestCaseResult> ret = std::make_shared<TestCaseResult>(info->GetDataCount(), EXECUTE_RESULT::UNKNOWN_ERROR, "");
  auto status = OnTestCaseFinished(pci, task, ret);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "FATAL ERROR");
    abort();
  }
}
}  // namespace
void ParallelRunTests(TestEnv& env, int p_models, size_t current_runs) {
  LOGF_DEFAULT(ERROR, "Running tests in parallel with %d threads\n", p_models);
  p_models = (int)std::min<size_t>(p_models, env.tests.size());
  env.next_test_to_run = p_models;
  for (int i = 0; i != p_models; ++i) {
    TestCaseTask* t = new TestCaseTask{env, i, current_runs};
    PTP_WORK work = CreateThreadpoolWork(RunTestCase, t, nullptr);
    if (!work) {
      delete t;
      throw std::runtime_error("schedule test task failed");
    }
    SubmitThreadpoolWork(work);
  }
  env.finished->wait();
}

void PTestRunner::Start(size_t concurrent_runs) {
  concurrent_runs = std::min<size_t>(std::max<size_t>(1, concurrent_runs), c_->GetDataCount());
  next_test_to_run = 0;
  for (size_t i = 0; i != concurrent_runs; ++i) {
    ScheduleNew();
  }
}

bool PTestRunner::ScheduleNew() {
  size_t next_test = next_test_to_run++;
  if (next_test >= c_->GetDataCount()) return false;
  DataTask* t = new DataTask{this, next_test};
  PTP_WORK work = CreateThreadpoolWork(RunSingleDataItem, t, nullptr);
  if (!work) {
    LOGF_DEFAULT(ERROR, "schedule test task failed\n");
    return false;
  }
  SubmitThreadpoolWork(work);
  return true;
}

void PTestRunner::OnTaskFinished(size_t, EXECUTE_RESULT, PTP_CALLBACK_INSTANCE pci) noexcept {
  try {
    ScheduleNew();
    if (++finished == c_->GetDataCount()) {
      //For each test case, only one DataTask can reach here
      //copy things out because we want to free DataTask before calling the callback
      finish(result, pci);
    }
  } catch (std::exception& ex) {
    LOGF_DEFAULT(ERROR, "%s:unrecoverable error:%s,exit...\n", c_->GetTestCaseName().c_str(), ex.what());
    abort();
  } catch (...) {
    LOGF_DEFAULT(ERROR, "%s:unrecoverable error,exit...\n", c_->GetTestCaseName().c_str());
    abort();
  }
}

PTestRunner::PTestRunner(std::shared_ptr<Lotus::InferenceSession> session1,
                         ITestCase* c,
                         TestCaseCallBack on_finished1) : DataRunner(session1, c->GetTestCaseName(), c, on_finished1), next_test_to_run(0), finished(0) {
}
