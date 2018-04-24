#include "testenv.h"
#include "runner.h"
#include <Windows.h>

using std::experimental::filesystem::v1::path;

extern std::vector<onnx::TensorProto> LoadTensors(const std::vector<path>& pb_files);
extern EXECUTE_RESULT ExecuteModelWithProtobufs(Lotus::InferenceSession& sess, const std::vector<onnx::TensorProto>& input_pbs,
                                                const std::vector<onnx::TensorProto>& output_pbs, const char* test_case_name,
                                                const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& input_value_info, Lotus::AllocatorManager& allocatorManager);

namespace {
struct TestCaseTask {
  TestEnv& env;
  const int task_id;
  const size_t concurrent_runs;
  std::vector<TestCaseResult>& results;
};

struct DataTask {
  RunContext* env;
  const size_t task_id;
};

void __stdcall RunTestCase(
    _Inout_ PTP_CALLBACK_INSTANCE,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work);

void OnTestCaseFinished(TestCaseTask* task, TestCaseResult& result) {
  task->results[task->task_id] = result;
  TestEnv& env = task->env;
  env.finished->onFinished(0);
  int next_test = env.next_test_to_run++;
  if (next_test < env.tests.size()) {
    TestCaseTask* t = new TestCaseTask{env, next_test, task->concurrent_runs, task->results};
    PTP_WORK work = CreateThreadpoolWork(RunTestCase, t, nullptr);
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

void OnDataTestFinished(DataTask* task, EXECUTE_RESULT result) {
  try {
    task->env->result.excution_result[task->task_id] = result;
    task->env->result.node_name = task->env->node_name;
    size_t next_test = task->env->next_test_to_run++;
    if (next_test < task->env->test_case.input_pb_files.size()) {
      DataTask* t = new DataTask{task->env, next_test};
      PTP_WORK work = CreateThreadpoolWork(RunTestCase, t, nullptr);
      if (!work) {
        printf("schedule test task failed\n");
        abort();
      }
      SubmitThreadpoolWork(work);
    }
    if (++task->env->finished == task->env->test_case.input_pb_files.size()) {
      task->env->on_finished(task->env->result);
      delete task->env;
    }
    delete task;
  } catch (std::exception& ex) {
    printf("%s:unrecoverable error:%s,exit...\n", task->env->test_case.test_case_name.c_str(), ex.what());
    abort();
  } catch (...) {
    printf("%s:unrecoverable error,exit...\n", task->env->test_case.test_case_name.c_str());
    abort();
  }
}

void __stdcall RunDataTest(
    _Inout_ PTP_CALLBACK_INSTANCE,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work) {
  CloseThreadpoolWork(work);
  DataTask* task((DataTask*)context);
  try {
    auto& test_case = task->env->test_case;
    const std::vector<path>& inputs = test_case.input_pb_files[task->task_id];
    const std::vector<path>& outputs = test_case.output_pb_files[task->task_id];
    std::vector<onnx::TensorProto> input_pbs = LoadTensors(inputs);
    std::vector<onnx::TensorProto> output_pbs = LoadTensors(outputs);
    OnDataTestFinished(task, ExecuteModelWithProtobufs(*task->env->session, input_pbs, output_pbs, test_case.test_case_name.c_str(), task->env->input_info, task->env->allocatorManager));
    return;
  } catch (std::exception& ex) {
    printf("%s:%s", task->env->test_case.test_case_name.c_str(), ex.what());
  } catch (...) {
    printf("%s:unknown error\n", task->env->test_case.test_case_name.c_str());
  }
  OnDataTestFinished(task, EXECUTE_RESULT::FAILED_TO_RUN);
}

void __stdcall RunTestCase(
    _Inout_ PTP_CALLBACK_INSTANCE,
    _Inout_opt_ PVOID context,
    _Inout_ PTP_WORK work) {
  CloseThreadpoolWork(work);
  TestCaseTask* task((TestCaseTask*)context);
  const TestCaseInfo& info = task->env.tests[task->task_id];
  try {
    RunSingleTestCase(task->env, task->task_id, task->concurrent_runs, [task](TestCaseResult& result) {
      OnTestCaseFinished(task, result);
    });
    return;
  } catch (std::exception& ex) {
    printf(ex.what());
  } catch (...) {
    printf("unknown error\n");
  }
  TestCaseResult ret{std::vector<EXECUTE_RESULT>(info.input_pb_files.size(), EXECUTE_RESULT::UNKNOWN_ERROR), ""};
  OnTestCaseFinished(task, ret);
}
}  // namespace
void ParallelRunTests(TestEnv& env, int p_models, size_t current_runs, std::vector<TestCaseResult>& results) {
  printf("Running tests in parallel with %d threads\n", p_models);
  p_models = (int)std::min<size_t>(p_models, env.tests.size());
  env.next_test_to_run = p_models;
  for (int i = 0; i != p_models; ++i) {
    TestCaseTask* t = new TestCaseTask{env, i, current_runs, results};
    PTP_WORK work = CreateThreadpoolWork(RunTestCase, t, nullptr);
    if (!work) {
      delete t;
      throw std::runtime_error("schedule test task failed");
    }
    SubmitThreadpoolWork(work);
  }
  env.finished->wait();
}

void ParallelRunData(RunContext* env, size_t concurrent_runs) {
  concurrent_runs = std::min<size_t>(concurrent_runs, env->test_case.input_pb_files.size());
  env->next_test_to_run = concurrent_runs;
  for (size_t i = 0; i != concurrent_runs; ++i) {
    DataTask* t = new DataTask{env, i};
    PTP_WORK work = CreateThreadpoolWork(RunDataTest, t, nullptr);
    if (!work) {
      delete t;
      throw std::runtime_error("schedule test task failed");
    }
    SubmitThreadpoolWork(work);
  }
}