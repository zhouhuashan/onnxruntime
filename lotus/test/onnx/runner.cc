#include "runner.h"
#include <core/common/logging/logging.h>
#include <core/framework/tensorprotoutils.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif
#ifdef _MSC_VER
#include <filesystem>
#endif
#include <fstream>
#include <cmath>
#include <core/common/logging/logging.h>
#include <core/framework/compare_mlvalue.h>
#include "TestCase.h"
#ifdef _WIN32
#include "FixedCountFinishCallbackWin.h"
#endif
using std::experimental::filesystem::v1::directory_iterator;
using std::experimental::filesystem::v1::is_directory;
using std::experimental::filesystem::v1::path;
using namespace Lotus;

#ifdef _WIN32
Lotus::Common::Status SetWindowsEvent(LOTUS_CALLBACK_INSTANCE pci, HANDLE finish_event) {
  if (pci)
    SetEventWhenCallbackReturns(pci, finish_event);
  else if (!SetEvent(finish_event)) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "SetEvent failed");
  }
  return Common::Status::OK();
}
#else
Lotus::Common::Status SetWindowsEvent(LOTUS_CALLBACK_INSTANCE, pthread_cond_t* finish_event) {
  if (!pthread_cond_broadcast(finish_event))
    return Common::Status::OK();
  else
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "SetEvent failed");
}
#endif

Lotus::Common::Status RunTests(TestEnv& env, int p_models, int concurrent_runs, size_t repeat_count) {
  TestResultStat& stat = env.stat;
  stat.total_model_count = env.tests.size();
  stat.total_test_case_count = std::accumulate(env.tests.begin(), env.tests.end(), static_cast<size_t>(0), [](size_t v, const ITestCase* info) {
    return info->GetDataCount() + v;
  });
  std::vector<std::shared_ptr<TestCaseResult>> results;
#ifdef _WIN32
  if (p_models > 1 && env.tests.size() > 1) {
    ParallelRunTests(env, p_models, concurrent_runs, repeat_count);
    results = env.finished->getResults();
  } else
#endif
  {
    //run models one by one
    for (size_t i = 0; i != env.tests.size(); ++i) {
      const char* test_case_name = env.tests[i]->GetTestCaseName().c_str();
      bool finished = false;
#ifdef _WIN32
      HANDLE finish_event = CreateEvent(
          NULL,                // default security attributes
          TRUE,                // manual-reset event
          FALSE,               // initial state is nonsignaled
          TEXT("FinishEvent")  // object name
      );
      if (finish_event == NULL) {
        return LOTUS_MAKE_STATUS(LOTUS, FAIL, "unable to create finish event");
      }
#else
      pthread_cond_t finish_event_data = PTHREAD_COND_INITIALIZER;
      pthread_cond_t* finish_event = &finish_event_data;
      pthread_mutex_t finish_event_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif
      RunSingleTestCase(env.tests[i], env.sf, concurrent_runs, repeat_count, nullptr, [repeat_count, &finished, &results, finish_event, concurrent_runs, test_case_name](std::shared_ptr<TestCaseResult> result, LOTUS_CALLBACK_INSTANCE pci) {
        //TODO:output this information to a xml
        if (concurrent_runs == 1) {
          TIME_SPEC ts = result->GetSpentTime();
          double spent = TimeSpecToSeconds(&ts);
          double spent2 = spent / result->GetExcutionResult().size() / repeat_count;
          LOGF_DEFAULT(ERROR, "Test %s finished in %.3g seconds, took %.3g for each input", test_case_name, spent, spent2);
        }
        results.push_back(result);
        finished = true;
        return SetWindowsEvent(pci, finish_event);
      });
#ifdef _WIN32
      DWORD dwWaitResult = WaitForSingleObject(finish_event, INFINITE);
      if (dwWaitResult != WAIT_OBJECT_0) {
        return LOTUS_MAKE_STATUS(LOTUS, FAIL, "WaitForSingleObject failed");
      }
#else
      pthread_mutex_lock(&finish_event_mutex);
      while (!finished) {
        pthread_cond_wait(finish_event, &finish_event_mutex);
      }
      pthread_mutex_unlock(&finish_event_mutex);
#endif
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

std::vector<ITestCase*> LoadTests(const std::vector<std::string>& input_paths, const std::vector<std::string>& whitelisted_test_cases, Lotus::AllocatorPtr allocator) {
  std::vector<ITestCase*> tests;
  std::vector<path> paths;
  for (const std::string& s : input_paths) {
    paths.push_back(s);
  }
  const path ext_onnx(".onnx");
  while (!paths.empty()) {
    path node_data_root_path = paths.back();
    paths.pop_back();
    for (directory_iterator test_case_dir(node_data_root_path), end; test_case_dir != end; ++test_case_dir) {
      if (is_directory(*test_case_dir)) {
        paths.push_back(test_case_dir->path());
        continue;
      }

      std::string filename = test_case_dir->path().filename().string();
      if (!test_case_dir->path().has_extension()) continue;
      if (test_case_dir->path().extension() != ext_onnx) continue;
      std::string test_case_name = test_case_dir->path().parent_path().filename().string();
      if (test_case_name.compare(0, 5, "test_") == 0) test_case_name = test_case_name.substr(5);
      if (!whitelisted_test_cases.empty() && std::find(whitelisted_test_cases.begin(), whitelisted_test_cases.end(), test_case_name) == whitelisted_test_cases.end()) {
        continue;
      }

      OnnxTestCase* l = new OnnxTestCase(allocator, test_case_name);
      auto status = l->SetModelPath(test_case_dir->path());
      if (!status.IsOK()) {
        std::string s = test_case_dir->path().string();
        LOGF_DEFAULT(ERROR, "load data from %s failed:%s\n", s.c_str(), status.ErrorMessage().c_str());
        delete l;
        continue;
      }
      tests.push_back(l);
    }
  }
  return tests;
}

SeqTestRunner::SeqTestRunner(std::shared_ptr<Lotus::InferenceSession> session1,
                             ITestCase* c, size_t repeat_count,
                             TestCaseCallBack on_finished1) : DataRunner(session1, c->GetTestCaseName(), c, on_finished1), repeat_count_(repeat_count) {
}

DataRunner::DataRunner(std::shared_ptr<Lotus::InferenceSession> session1, const std::string& test_case_name1, ITestCase* c, TestCaseCallBack on_finished1) : session(session1), test_case_name_(test_case_name1), c_(c), on_finished(on_finished1) {
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
  std::unordered_map<std::string, Lotus::MLValue> feeds;
  std::vector<Lotus::MLValue> output_values;
  Common::Status status = c_->LoadInputData(task_id, feeds);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "%s", status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }
  std::vector<MLValue> p_fetches;
  TIME_SPEC start_time, end_time;
  GetMonotonicTimeCounter(&start_time);
  status = session->Run(feeds, &p_fetches);
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
      LOGF_DEFAULT(ERROR, "%s:%s", test_case_name_.c_str(), ret.second.c_str());
    }
    if (compare_result != COMPARE_RESULT::SUCCESS) {
      break;
    }
  }
  return res;
}

void SeqTestRunner::Start(size_t) {
  const size_t data_count = c_->GetDataCount();
  for (size_t j = 0; j != repeat_count_; ++j)
    for (size_t i = 0; i != data_count; ++i) {
      RunTask(i, nullptr, i == 0);
    }
  finish(result, nullptr);
}

void RunSingleTestCase(ITestCase* info, const SessionFactory& sf, size_t concurrent_runs, size_t repeat_count, LOTUS_CALLBACK_INSTANCE pci, TestCaseCallBack on_finished) {
  std::shared_ptr<TestCaseResult> ret;
  size_t data_count = info->GetDataCount();
  {
    DataRunner* r = nullptr;
    std::string node_name;
    Lotus::Common::Status status = info->GetNodeName(&node_name);
    if (!status.IsOK()) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), status.ErrorMessage().c_str());
      ret = std::make_shared<TestCaseResult>(data_count, StatusCodeToExecuteResult(status.Code()), node_name);
      goto end;
    }
    std::shared_ptr<Lotus::InferenceSession> session_object;
    try {
      status = sf.create(session_object, info->GetModelUrl(), info->GetTestCaseName());
      if (!status.IsOK()) {
        LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), status.ErrorMessage().c_str());
        ret = std::make_shared<TestCaseResult>(data_count, StatusCodeToExecuteResult(status.Code()), node_name);
        goto end;
      }
    } catch (Lotus::NotImplementedException& ex) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), ex.what());
      ret = std::make_shared<TestCaseResult>(data_count, EXECUTE_RESULT::NOT_SUPPORT, node_name);
      goto end;
    } catch (std::exception& ex) {
      LOGF_DEFAULT(ERROR, "load model %s failed:%s\n", info->GetTestCaseName().c_str(), ex.what());
      ret = std::make_shared<TestCaseResult>(data_count, EXECUTE_RESULT::LOAD_MODEL_FAILED, node_name);
      goto end;
    }
    LOGF_DEFAULT(INFO, "testing %s\n", info->GetTestCaseName().c_str());
#ifdef _WIN32
    if (concurrent_runs > 1 && data_count > 1) {
      r = new PTestRunner(session_object, info, on_finished);
    } else
#endif
    {
      r = new SeqTestRunner(session_object, info, repeat_count, on_finished);
    }
    r->Start(concurrent_runs);
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
