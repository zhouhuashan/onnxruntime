#pragma once
#include <atomic>
#include <vector>
#include "TestResultStat.h"
#include <core/common/common.h>
#include <core/framework/inference_session.h>
#include "test/framework/TestAllocatorManager.h"

#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif

class ITestCase;
class TestCaseResult;
template <typename T>
class FixedCountFinishCallbackImpl;
using FixedCountFinishCallback = FixedCountFinishCallbackImpl<TestCaseResult>;

class SessionFactory {
 private:
  const std::string provider_;
  bool enable_mem_pattern_ = true;
  bool enable_cpu_mem_arena_ = true;

 public:
  SessionFactory(const std::string& provider, bool enable_mem_pattern, bool enable_cpu_mem_arena) : provider_(provider), enable_mem_pattern_(enable_mem_pattern), enable_cpu_mem_arena_(enable_cpu_mem_arena) {}
  //Create an initialized session from a given model url
  Lotus::Common::Status create(std::shared_ptr<Lotus::InferenceSession>& sess, const std::experimental::filesystem::v1::path& model_url, const std::string& logid) const;
};

class TestEnv {
 public:
  std::vector<ITestCase*> tests;
  std::atomic_int next_test_to_run;
  TestResultStat& stat;
  FixedCountFinishCallback* finished;
  const SessionFactory& sf;
  TestEnv(const std::vector<ITestCase*>& tests, TestResultStat& stat1, SessionFactory& sf1);
  ~TestEnv();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TestEnv);
};
