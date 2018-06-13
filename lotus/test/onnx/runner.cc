#include "runner.h"
#include <core/common/logging/logging.h>
#include <core/framework/tensorprotoutils.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef _MSC_VER
#include <filesystem>
#endif
#include <fstream>
#include <cmath>
#include <core/common/logging/logging.h>

#include "FixedCountFinishCallback.h"
#include "TestCase.h"
#include "CompareMLValue.h"

using std::experimental::filesystem::v1::directory_iterator;
using std::experimental::filesystem::v1::is_directory;
using std::experimental::filesystem::v1::path;
using namespace Lotus;

EXECUTE_RESULT DataRunner::ExecuteSingleData(const std::unordered_map<std::string, Lotus::MLValue>& feeds, const std::vector<Lotus::MLValue>& output_values) {
  std::vector<MLValue> p_fetches;
  TIME_SPEC start_time, end_time;
  GetMonotonicTimeCounter(&start_time);
  Common::Status status = session->Run(feeds, &p_fetches);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "%s:%s\n", test_case_name_.c_str(), status.ErrorMessage().c_str());
    return StatusCodeToExecuteResult(status.Code());
  }
  GetMonotonicTimeCounter(&end_time);
  AccumulateTimeSpec(&spent_time_, &start_time, &end_time);

  for (size_t i = 0; i != output_values.size(); ++i) {
    const MLValue& o = p_fetches.at(i);
    //TODO: why queue id is zero?
    if (o.Fence())
      o.Fence()->BeforeUsingAsInput(LotusIR::kCpuExecutionProvider, 0);
    const MLValue& e = output_values.at(i);
    std::pair<EXECUTE_RESULT, size_t> ret = compareMLValue(o, e);
    EXECUTE_RESULT compare_result = ret.first;
    switch (compare_result) {
      case EXECUTE_RESULT::RESULT_DIFFERS:
        LOGF_DEFAULT(ERROR, "%s: the %zd-th value of the %zd-th output differs\n", test_case_name_.c_str(), ret.second, i);
        break;
      case EXECUTE_RESULT::SHAPE_MISMATCH:
        LOGF_DEFAULT(ERROR, "%s: shape mismatch for the %zd-th output\n", test_case_name_.c_str(), i);
        break;
      case EXECUTE_RESULT::TYPE_MISMATCH:
        LOGF_DEFAULT(ERROR, "%s: type mismatch for the %zd-th output\n", test_case_name_.c_str(), i);
    }
    if (compare_result != EXECUTE_RESULT::SUCCESS)
      return compare_result;
  }
  LOGF_DEFAULT(ERROR, "test %s succeeded\n", test_case_name_.c_str());
  return EXECUTE_RESULT::SUCCESS;
}

void RunTests(TestEnv& env, int p_models, int concurrent_runs) {
  TestResultStat& stat = env.stat;
  stat.total_test_case_count = std::accumulate(env.tests.begin(), env.tests.end(), static_cast<size_t>(0), [](size_t v, const ITestCase* info) {
    return info->GetDataCount() + v;
  });
  std::vector<std::shared_ptr<TestCaseResult>> results;
#ifdef _WIN32
  if (p_models > 1 && env.tests.size() > 1) {
    ParallelRunTests(env, p_models, concurrent_runs);
    results = env.finished->getResults();
  } else
#endif
  {
    //run models one by one
    FixedCountFinishCallback c(static_cast<int>(env.tests.size()));
    for (size_t i = 0; i != env.tests.size(); ++i) {
      RunSingleTestCase(env, i, concurrent_runs, [i, &c](std::shared_ptr<TestCaseResult> result) {
        TIME_SPEC ts = result->GetSpentTime();
        double spent = TimeSpecToSeconds(&ts);
        //TODO:output this information to a xml
        LOGF_DEFAULT(ERROR, "Test %zd finished in %.3g seconds", i, spent);
        c.onFinished(i, result);
      });
    }
    c.wait();
    results = c.getResults();
  }
  for (size_t i = 0; i != env.tests.size(); ++i) {
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
        case EXECUTE_RESULT::SHAPE_MISMATCH:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
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
          abort();
      }
    }
  }
}

std::vector<ITestCase*> LoadTests(const std::vector<string>& input_paths, const std::vector<std::string>& whitelisted_test_cases, Lotus::AllocatorPtr allocator) {
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
      tests.emplace_back(l);
    }
  }
  return tests;
}

SeqTestRunner::SeqTestRunner(std::shared_ptr<Lotus::InferenceSession> session1,
                             ITestCase* c,
                             std::function<void(std::shared_ptr<TestCaseResult> result)> on_finished1) : DataRunner(session1, c->GetTestCaseName(), c, on_finished1) {
}

DataRunner::DataRunner(std::shared_ptr<Lotus::InferenceSession> session1, const std::string& test_case_name1, ITestCase* c, std::function<void(std::shared_ptr<TestCaseResult> result)> on_finished1) : session(session1), test_case_name_(test_case_name1), c_(c), on_finished(on_finished1), result(std::make_shared<TestCaseResult>(c->GetDataCount(), EXECUTE_RESULT::UNKNOWN_ERROR, c->GetNodeName())) {
  SetTimeSpecToZero(&spent_time_);
}

void DataRunner::RunTask(size_t task_id) {
  try {
    RunTaskImpl(task_id);
    return;
  } catch (std::exception& ex) {
    LOGF_DEFAULT(ERROR, "%s:%s", c_->GetTestCaseName().c_str(), ex.what());
  }
  SetResult(task_id, EXECUTE_RESULT::WITH_EXCEPTION);
}

void DataRunner::RunTaskImpl(size_t task_id) {
  std::unordered_map<std::string, Lotus::MLValue> feeds;
  std::vector<Lotus::MLValue> output_values;
  auto status = c_->LoadDataPair(task_id, feeds, output_values);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "%s", status.ErrorMessage().c_str());
    SetResult(task_id, StatusCodeToExecuteResult(status.Code()));
    return;
  }
  EXECUTE_RESULT r = ExecuteSingleData(feeds, output_values);
  SetResult(task_id, r);
}

void SeqTestRunner::Start(size_t) {
  const size_t data_count = c_->GetDataCount();
  std::string node_name = c_->GetNodeName();
  for (size_t i = 0; i != data_count; ++i) {
    RunTask(i);
  }
  finish(result);
}

void RunSingleTestCase(TestEnv& env, size_t test_index, size_t concurrent_runs, std::function<void(std::shared_ptr<TestCaseResult> result)> on_finished) {
  std::shared_ptr<TestCaseResult> ret;
  ITestCase* info = env.tests[test_index];
  size_t data_count = info->GetDataCount();
  {
    DataRunner* r = nullptr;
    std::string node_name = info->GetNodeName();
    std::shared_ptr<Lotus::InferenceSession> session_object;
    try {
      Lotus::Common::Status status = env.sf.create(session_object, info->GetModelUrl(), info->GetTestCaseName());
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
      r = new SeqTestRunner(session_object, info, on_finished);
    }
    r->Start(concurrent_runs);
    return;
  }
end:
  on_finished(ret);
}

EXECUTE_RESULT StatusCodeToExecuteResult(int input) {
  switch (input) {
    case StatusCode::NOT_IMPLEMENTED:
      return EXECUTE_RESULT::NOT_SUPPORT;
    case StatusCode::INVALID_GRAPH:
      return EXECUTE_RESULT::INVALID_GRAPH;
    case StatusCode::INVALID_ARGUMENT:
      return EXECUTE_RESULT::INVALID_ARGUMENT;
    default:
      return EXECUTE_RESULT::UNKNOWN_ERROR;
  }
}

void DataRunner::SetResult(size_t task_id, EXECUTE_RESULT res) noexcept {
  result->SetResult(task_id, res);
  OnTaskFinished(task_id, res);
}
