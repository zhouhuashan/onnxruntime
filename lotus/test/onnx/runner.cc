#include "runner.h"

#include <fstream>
#include <cmath>

#include <core/common/logging/logging.h>
#include <core/graph/constants.h>
#include <core/platform/env.h>
#include <core/framework/tensorprotoutils.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#include <unsupported/Eigen/CXX11/ThreadPool>
#endif
#include <test/compare_mlvalue.h>
#include "TestCase.h"
#include "FixedCountFinishCallback.h"
using std::experimental::filesystem::v1::directory_iterator;
using std::experimental::filesystem::v1::is_directory;
using std::experimental::filesystem::v1::path;
using namespace Lotus;
using ::Lotus::Common::Status;

void LOTUS_CALLBACK RunTestCase(LOTUS_CALLBACK_INSTANCE pci, void* context, LOTUS_WORK work) {
  LotusCloseThreadpoolWork(work);
  TestCaseTask* task((TestCaseTask*)context);
  ITestCase* info = task->env.tests[task->task_id];
  std::shared_ptr<TestCaseResult> ret;
  try {
    RunSingleTestCase(info, task->env.sf, task->concurrent_runs, task->repeat_count, task->pool, pci, [task](std::shared_ptr<TestCaseResult> result, LOTUS_CALLBACK_INSTANCE pci) {
      return OnTestCaseFinished(pci, task, result);
    });
    return;
  } catch (std::exception& ex) {
    ret = std::make_shared<TestCaseResult>(info->GetDataCount(), EXECUTE_RESULT::WITH_EXCEPTION, ex.what());
  }
  auto status = OnTestCaseFinished(pci, task, ret);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "FATAL ERROR");
    abort();
  }
}

void PTestRunner::Start(LOTUS_CALLBACK_INSTANCE, size_t concurrent_runs) {
  concurrent_runs = std::min<size_t>(std::max<size_t>(1, concurrent_runs), c_->GetDataCount());
  next_test_to_run = 0;
  for (size_t i = 0; i != concurrent_runs; ++i) {
    if (!ScheduleNew()) {
      throw std::runtime_error("ScheduleNew task failed");
    }
  }
}

bool PTestRunner::ScheduleNew() {
  size_t next_test = next_test_to_run++;
  if (next_test >= c_->GetDataCount()) return false;
  DataTask* t = new DataTask{this, next_test};
  Status st = CreateAndSubmitThreadpoolWork(RunSingleDataItem, t, tpool_);
  if (!st.IsOK()) {
    delete t;
    LOGF_DEFAULT(ERROR, "schedule test task failed: %s\n", st.ErrorMessage().c_str());
    return false;
  }
  return true;
}

void PTestRunner::OnTaskFinished(size_t, EXECUTE_RESULT, LOTUS_CALLBACK_INSTANCE pci) noexcept {
  try {
    ScheduleNew();
    if (++finished == c_->GetDataCount()) {
      //For each test case, only one DataTask can reach here
      finish(pci);
    }
  } catch (std::exception& ex) {
    LOGF_DEFAULT(ERROR, "%s:unrecoverable error:%s,exit...\n", c_->GetTestCaseName().c_str(), ex.what());
    abort();
  } catch (...) {
    LOGF_DEFAULT(ERROR, "%s:unrecoverable error,exit...\n", c_->GetTestCaseName().c_str());
    abort();
  }
}

PTestRunner::PTestRunner(std::shared_ptr<::Lotus::InferenceSession> session1,
                         ITestCase* c, PThreadPool tpool,
                         TestCaseCallBack on_finished1) : DataRunner(session1, c->GetTestCaseName(), c, on_finished1), next_test_to_run(0), finished(0), tpool_(tpool) {
}

void LOTUS_CALLBACK RunSingleDataItem(LOTUS_CALLBACK_INSTANCE instance, void* context, LOTUS_WORK work) {
  LotusCloseThreadpoolWork(work);
  DataTask* task((DataTask*)context);
  PTestRunner* env = task->env;
  const size_t task_id = task->task_id;
  delete task;
  env->RunTask(task_id, instance, true);
}

Status OnTestCaseFinished(LOTUS_CALLBACK_INSTANCE pci, TestCaseTask* task, std::shared_ptr<TestCaseResult> result) {
  FixedCountFinishCallback* finished = task->env.finished;
  auto task_id = task->task_id;
  bool failed = false;
  {
    std::unique_ptr<TestCaseTask> unused(task);
    TestEnv& env = task->env;
    int next_test = env.next_test_to_run++;
    if (static_cast<size_t>(next_test) < env.tests.size()) {
      //schedule the next TestCase
      std::unique_ptr<TestCaseTask> t(new TestCaseTask{env, next_test, task->concurrent_runs, task->repeat_count, task->pool});
      Status st = CreateAndSubmitThreadpoolWork(RunTestCase, t.get(), task->pool);
      if (st.IsOK()) {
        t.release();
      } else
        return st;
    }
  }
  if (failed)
    return finished->fail(pci);
  else
    return finished->onFinished(task_id, result, pci);
}

//Do not run this function in the thread pool passed in
static Status ParallelRunTests(TestEnv& env, int p_models, size_t current_runs, size_t repeat_count, PThreadPool pool) {
  p_models = (int)std::min<size_t>(p_models, env.tests.size());
  LOGF_DEFAULT(ERROR, "Running tests in parallel: at most %d models at any time", p_models);
  env.next_test_to_run = p_models;
  for (int i = 0; i != p_models; ++i) {
    std::unique_ptr<TestCaseTask> t(new TestCaseTask{env, i, current_runs, repeat_count, pool});
    auto st = CreateAndSubmitThreadpoolWork(RunTestCase, t.get(), pool);
    if (!st.IsOK()) return st;
    t.release();
  }
  bool ret = env.finished->wait();
  if (!ret) {
    return Status(::Lotus::Common::LOTUS, ::Lotus::Common::FAIL, "ParallelRunTests failed");
  }
  LOGF_DEFAULT(ERROR, "Running tests finished. Generating report");
  return Status::OK();
}

Status RunTests(TestEnv& env, int p_models, int concurrent_runs, size_t repeat_count, PThreadPool tpool) {
  TestResultStat& stat = env.stat;
  stat.total_model_count = env.tests.size();
  stat.total_test_case_count = std::accumulate(env.tests.begin(), env.tests.end(), static_cast<size_t>(0), [](size_t v, const ITestCase* info) {
    return info->GetDataCount() + v;
  });
  std::vector<std::shared_ptr<TestCaseResult>> results;
  if (p_models > 1 && env.tests.size() > 1) {
    LOTUS_RETURN_IF_ERROR(ParallelRunTests(env, p_models, concurrent_runs, repeat_count, tpool));
    results = env.finished->getResults();
  } else {
    //run models one by one
    for (size_t i = 0; i != env.tests.size(); ++i) {
      const char* test_case_name = env.tests[i]->GetTestCaseName().c_str();
      LOTUS_EVENT ev;
      LOTUS_RETURN_IF_ERROR(CreateLotusEvent(&ev));
      RunSingleTestCase(env.tests[i], env.sf, concurrent_runs, repeat_count, tpool, nullptr, [repeat_count, &results, ev, concurrent_runs, test_case_name](std::shared_ptr<TestCaseResult> result, LOTUS_CALLBACK_INSTANCE pci) {
        //TODO:output this information to a xml
        if (concurrent_runs == 1) {
          TIME_SPEC ts = result->GetSpentTime();
          double spent = TimeSpecToSeconds(&ts);
          double spent2 = spent / result->GetExcutionResult().size() / repeat_count;
          LOGF_DEFAULT(ERROR, "Test %s finished in %.3g seconds, took %.3g for each input", test_case_name, spent, spent2);
        }
        results.push_back(result);
        return LotusSetEventWhenCallbackReturns(pci, ev);
      });
      LOTUS_RETURN_IF_ERROR(WaitAndCloseEvent(ev));
    }
  }
  for (size_t i = 0; i != env.tests.size(); ++i) {
    if (!results[i]) {
      stat.AddFailedTest(env.tests[i]->GetTestCaseName());
      continue;
    }
    const TestCaseResult& r = *results[i];
    for (const EXECUTE_RESULT res : r.GetExcutionResult()) {
      if (res != EXECUTE_RESULT::SUCCESS && res != EXECUTE_RESULT::NOT_SUPPORT) {
        stat.AddFailedTest(env.tests[i]->GetTestCaseName());
      }
      switch (res) {
        case EXECUTE_RESULT::SUCCESS:
          stat.succeeded++;
          break;
        case EXECUTE_RESULT::INVALID_ARGUMENT:
        case EXECUTE_RESULT::UNKNOWN_ERROR:
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::INVALID_GRAPH:
          stat.invalid_graph++;
          break;
        case EXECUTE_RESULT::WITH_EXCEPTION:
          stat.throwed_exception++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::RESULT_DIFFERS:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
        case EXECUTE_RESULT::SHAPE_MISMATCH:
        case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
        case EXECUTE_RESULT::TYPE_MISMATCH:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::NOT_SUPPORT:
          stat.not_implemented++;
          if (!r.node_name.empty()) stat.AddNotImplementedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::LOAD_MODEL_FAILED:
          stat.load_model_failed++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        default:
          return LOTUS_MAKE_STATUS(LOTUS, FAIL, "unknown result");
      }
    }
  }
  return Common::Status::OK();
}

std::vector<ITestCase*> LoadTests(const std::vector<path>& input_paths, const std::vector<std::string>& whitelisted_test_cases, ::Lotus::AllocatorPtr allocator) {
  std::vector<ITestCase*> tests;
  std::vector<path> paths(input_paths);
  const path ext_onnx(".onnx");
  while (!paths.empty()) {
    path node_data_root_path = paths.back();
    paths.pop_back();
    for (directory_iterator file_entry(node_data_root_path), end; file_entry != end; ++file_entry) {
      if (is_directory(*file_entry)) {
        paths.push_back(file_entry->path());
        continue;
      }

      if (!file_entry->path().has_extension()) continue;
      if (file_entry->path().extension() != ext_onnx) continue;
      std::string test_case_name = file_entry->path().parent_path().filename().string();
      if (test_case_name.compare(0, 5, "test_") == 0) test_case_name = test_case_name.substr(5);
      if (!whitelisted_test_cases.empty() && std::find(whitelisted_test_cases.begin(), whitelisted_test_cases.end(), test_case_name) == whitelisted_test_cases.end()) {
        continue;
      }

      ITestCase* l = CreateOnnxTestCase(allocator, test_case_name);
      auto status = l->SetModelPath(file_entry->path());
      if (!status.IsOK()) {
        std::string s = file_entry->path().string();
        LOGF_DEFAULT(ERROR, "load data from %s failed:%s\n", s.c_str(), status.ErrorMessage().c_str());
        delete l;
        continue;
      }
      tests.push_back(l);
    }
  }
  return tests;
}

SeqTestRunner::SeqTestRunner(std::shared_ptr<::Lotus::InferenceSession> session1,
                             ITestCase* c, size_t repeat_count,
                             TestCaseCallBack on_finished1) : DataRunner(session1, c->GetTestCaseName(), c, on_finished1), repeat_count_(repeat_count) {
}

DataRunner::DataRunner(std::shared_ptr<::Lotus::InferenceSession> session1, const std::string& test_case_name1, ITestCase* c, TestCaseCallBack on_finished1) : test_case_name_(test_case_name1), c_(c), session(session1), on_finished(on_finished1) {
  std::string s;
  c->GetNodeName(&s);
  result = std::make_shared<TestCaseResult>(c->GetDataCount(), EXECUTE_RESULT::UNKNOWN_ERROR, s);
  SetTimeSpecToZero(&spent_time_);
}

void DataRunner::RunTask(size_t task_id, LOTUS_CALLBACK_INSTANCE pci, bool store_result) {
  EXECUTE_RESULT res = EXECUTE_RESULT::UNKNOWN_ERROR;
  try {
    res = RunTaskImpl(task_id);
  } catch (std::exception& ex) {
    res = EXECUTE_RESULT::WITH_EXCEPTION;
    LOGF_DEFAULT(ERROR, "%s:%s", c_->GetTestCaseName().c_str(), ex.what());
  }
  if (store_result) {
    result->SetResult(task_id, res);
  }
  OnTaskFinished(task_id, res, pci);
}

EXECUTE_RESULT DataRunner::RunTaskImpl(size_t task_id) {
  std::unordered_map<std::string, ::Lotus::MLValue> feeds;
  std::vector<::Lotus::MLValue> output_values;
  Common::Status status = c_->LoadInputData(task_id, feeds);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "%s", status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }

  // Create output feed
  std::vector<std::string> output_names;
  for (auto const& outp : *(session->GetModelOutputs().second)) {
    output_names.push_back(outp->Name());
  }

  RunOptions run_options;
  std::vector<MLValue> p_fetches;
  TIME_SPEC start_time, end_time;
  GetMonotonicTimeCounter(&start_time);
  status = session->Run(run_options, feeds, output_names, &p_fetches);
  GetMonotonicTimeCounter(&end_time);
  AccumulateTimeSpec(&spent_time_, &start_time, &end_time);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "%s:%s\n", test_case_name_.c_str(), status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }
  //TODO: if there are no output value files, just skip the validation
  status = c_->LoadOutputData(task_id, output_values);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "%s", status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }

  double per_sample_tolerance, relative_per_sample_tolerance;
  bool post_procesing;

  if (!(status = c_->GetPerSampleTolerance(&per_sample_tolerance)).IsOK()) {
    LOGF_DEFAULT(ERROR, "%s", status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }
  if (!(status = c_->GetRelativePerSampleTolerance(&relative_per_sample_tolerance)).IsOK()) {
    LOGF_DEFAULT(ERROR, "%s", status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }
  if (!(status = c_->GetPostProcessing(&post_procesing)).IsOK()) {
    LOGF_DEFAULT(ERROR, "%s", status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }
  EXECUTE_RESULT res = EXECUTE_RESULT::SUCCESS;
  for (size_t i = 0; i != output_values.size(); ++i) {
    const MLValue& o = p_fetches.at(i);
    //this is the default value for provider sync.Currently only one execution queue for CPU.
    int queue_id = 0;
    if (o.Fence())
      o.Fence()->BeforeUsingAsInput(LotusIR::kCpuExecutionProvider, queue_id);
    const onnx::ValueInfoProto& v = c_->GetOutputInfoFromModel(i);
    std::pair<COMPARE_RESULT, std::string> ret = CompareMLValue(o, output_values.at(i), per_sample_tolerance, relative_per_sample_tolerance, post_procesing);
    COMPARE_RESULT compare_result = ret.first;
    if (compare_result == COMPARE_RESULT::SUCCESS) {
      ret = VerifyValueInfo(v, o);
      compare_result = ret.first;
      if (compare_result != COMPARE_RESULT::SUCCESS) {
        switch (compare_result) {
          case COMPARE_RESULT::NOT_SUPPORT:
            res = EXECUTE_RESULT::NOT_SUPPORT;
            break;
          case COMPARE_RESULT::SHAPE_MISMATCH:
            res = EXECUTE_RESULT::MODEL_SHAPE_MISMATCH;
            break;
          case COMPARE_RESULT::TYPE_MISMATCH:
            res = EXECUTE_RESULT::MODEL_TYPE_MISMATCH;
            break;
          default:
            res = EXECUTE_RESULT::UNKNOWN_ERROR;
        }
      }
    } else {
      switch (compare_result) {
        case COMPARE_RESULT::NOT_SUPPORT:
          res = EXECUTE_RESULT::NOT_SUPPORT;
          break;
        case COMPARE_RESULT::RESULT_DIFFERS:
          res = EXECUTE_RESULT::RESULT_DIFFERS;
          break;
        case COMPARE_RESULT::SHAPE_MISMATCH:
          res = EXECUTE_RESULT::SHAPE_MISMATCH;
          break;
        case COMPARE_RESULT::TYPE_MISMATCH:
          res = EXECUTE_RESULT::TYPE_MISMATCH;
          break;
        default:
          res = EXECUTE_RESULT::UNKNOWN_ERROR;
      }
    }
    if (compare_result != COMPARE_RESULT::SUCCESS && !ret.second.empty()) {
      c_->GetDatasetDebugInfoString(i);
      LOGF_DEFAULT(ERROR, "%s:%s", test_case_name_.c_str(), ret.second.c_str());
    }
    if (compare_result != COMPARE_RESULT::SUCCESS) {
      break;
    }
  }
  return res;
}

void SeqTestRunner::Start(LOTUS_CALLBACK_INSTANCE pci, size_t) {
  const size_t data_count = c_->GetDataCount();
  for (size_t idx_repeat = 0; idx_repeat != repeat_count_; ++idx_repeat)
    for (size_t idx_data = 0; idx_data != data_count; ++idx_data) {
      RunTask(idx_data, nullptr, idx_repeat == 0);
    }
  finish(pci);
}

void RunSingleTestCase(ITestCase* info, const SessionFactory& sf, size_t concurrent_runs, size_t repeat_count, PThreadPool tpool, LOTUS_CALLBACK_INSTANCE pci, TestCaseCallBack on_finished) {
  std::shared_ptr<TestCaseResult> ret;
  size_t data_count = info->GetDataCount();
  {
    DataRunner* r = nullptr;
    std::string node_name;
    Status status = info->GetNodeName(&node_name);
    if (!status.IsOK()) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), status.ErrorMessage().c_str());
      ret = std::make_shared<TestCaseResult>(data_count, StatusCodeToExecuteResult(status.Code()), node_name);
      goto end;
    }
    std::shared_ptr<::Lotus::InferenceSession> session_object;
    try {
      status = sf.create(session_object, info->GetModelUrl(), info->GetTestCaseName());
      if (!status.IsOK()) {
        LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), status.ErrorMessage().c_str());
        ret = std::make_shared<TestCaseResult>(data_count, StatusCodeToExecuteResult(status.Code()), node_name);
        goto end;
      }
    } catch (::Lotus::NotImplementedException& ex) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), ex.what());
      ret = std::make_shared<TestCaseResult>(data_count, EXECUTE_RESULT::NOT_SUPPORT, node_name);
      goto end;
    } catch (std::exception& ex) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), ex.what());
      ret = std::make_shared<TestCaseResult>(data_count, EXECUTE_RESULT::LOAD_MODEL_FAILED, node_name);
      goto end;
    }
    LOGF_DEFAULT(INFO, "testing %s\n", info->GetTestCaseName().c_str());
    if (concurrent_runs > 1 && data_count > 1) {
      r = new PTestRunner(session_object, info, tpool, on_finished);
    } else {
      r = new SeqTestRunner(session_object, info, repeat_count, on_finished);
    }
    r->Start(pci, concurrent_runs);
    return;
  }
end:
  on_finished(ret, pci);
}

EXECUTE_RESULT StatusCodeToExecuteResult(int input) {
  switch (input) {
    case Common::NOT_IMPLEMENTED:
      return EXECUTE_RESULT::NOT_SUPPORT;
    case Common::INVALID_GRAPH:
      return EXECUTE_RESULT::INVALID_GRAPH;
    case Common::INVALID_ARGUMENT:
      return EXECUTE_RESULT::INVALID_ARGUMENT;
    default:
      return EXECUTE_RESULT::UNKNOWN_ERROR;
  }
}
