#include "testenv.h"
#include "runner.h"
#include <Windows.h>

namespace {
struct Task {
  TestEnv& env;
  const int task_id;
  std::vector<TestCaseResult>& results;
  Task(TestEnv& env1,
       int task_id1, std::vector<TestCaseResult>& results1) : env(env1), task_id(task_id1), results(results1) {
  }
};

void __stdcall RunTest(
    _Inout_ PTP_CALLBACK_INSTANCE,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work);

void OnTestCaseFinished(Task* task, TestCaseResult& result) {
  task->results[task->task_id] = result;
  TestEnv& env = task->env;
  env.finished->onFinished(0);
  int next_test = env.next_test_to_run++;
  if (next_test < env.tests.size()) {
    Task* t = new Task(env, next_test, task->results);
    PTP_WORK work = CreateThreadpoolWork(RunTest, t, nullptr);
    if (!work) {
      printf("schedule test task failed\n");
      env.finished->onFinished(-1);
      delete task;
      return;
    }
    SubmitThreadpoolWork(work);
  }
  delete task;
}

void __stdcall RunTest(
    _Inout_ PTP_CALLBACK_INSTANCE,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work) {
  CloseThreadpoolWork(work);
  Task* task((Task*)context);
  const TestCaseInfo& info = task->env.tests[task->task_id];
  TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::UNKNOWN_ERROR), ""};
  try {
    RunSingleTestCase(task->env, task->task_id, [task](TestCaseResult& result) {
      OnTestCaseFinished(task, result);
    });
  } catch (std::exception& ex) {
    printf(ex.what());
    OnTestCaseFinished(task, ret);
  } catch (...) {
    printf("unknown error\n");
    OnTestCaseFinished(task, ret);
  }
}
}  // namespace
void ParallelRunTests(TestEnv& env, int p_models, std::vector<TestCaseResult>& results) {
  printf("Running tests in parallel with %d threads\n", p_models);
  p_models = (int)std::min<size_t>(p_models, env.tests.size());
  env.next_test_to_run = p_models;
  for (int i = 0; i != p_models; ++i) {
    Task* t = new Task(env, i, results);
    PTP_WORK work = CreateThreadpoolWork(RunTest, t, nullptr);
    if (!work) {
      delete t;
      throw std::runtime_error("schedule test task failed");
    }
    SubmitThreadpoolWork(work);
  }
  env.finished->wait();
}