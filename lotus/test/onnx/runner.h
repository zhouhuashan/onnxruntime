#pragma once
#include <vector>
#include <string>
#include "testenv.h"

namespace Lotus {
class AllocatorManager;
}
struct TestCaseResult {
  std::vector<EXECUTE_RESULT> excution_result;
  //only valid for single node tests;
  std::string node_name;
};

struct RunContext {
  const TestCaseInfo& test_case;
  //only valid for single node tests;
  std::string node_name;
  std::shared_ptr<Lotus::InferenceSession> session;
  const google::protobuf::RepeatedPtrField< ::ONNX_NAMESPACE::ValueInfoProto> input_info;
  Lotus::AllocatorManager& allocatorManager;
  std::function<void(TestCaseResult& result)> on_finished;
  std::atomic<size_t> next_test_to_run;
  std::atomic<size_t> finished;
  TestCaseResult result;
  RunContext(const TestCaseInfo& test_case1, const std::string& node_name1, std::shared_ptr<Lotus::InferenceSession> session1,
             const google::protobuf::RepeatedPtrField< ::ONNX_NAMESPACE::ValueInfoProto>& input_info1, Lotus::AllocatorManager& allocatorManager1,
             std::function<void(TestCaseResult& result)> on_finished1);
};
void RunSingleTestCase(TestEnv& env, size_t test_index, size_t concurrent_runs, std::function<void(TestCaseResult& result)> on_finished);
std::vector<TestCaseInfo> LoadTests(const std::vector<std::string>& input_paths, const std::vector<std::string>& whitelisted_test_cases);
void RunTests(TestEnv& env, int p_models, int concurrent_runs);

#ifdef _WIN32
extern void ParallelRunTests(TestEnv& env, int p_models, size_t concurrent_runs, std::vector<TestCaseResult>& results);
extern void ParallelRunData(std::shared_ptr<RunContext> env, size_t concurrent_runs);
#endif