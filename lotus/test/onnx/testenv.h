#pragma once
#include <atomic>
#include <vector>
#include "TestResultStat.h"
#include "FixedCountFinishCallback.h"
#include <core/common/common.h>
#include <core/framework/inference_session.h>
#include "test/framework/TestAllocatorManager.h"

class ITestCase;

class SessionFactory {
 private:
  const std::string provider;

 public:
  SessionFactory(const std::string& provider1) : provider(provider1) {}
  //Create an initialized session from a given model url
  Lotus::Common::Status create(std::shared_ptr<Lotus::InferenceSession>& sess, const std::experimental::filesystem::v1::path& model_url, const std::string& logid) const;
};

class TestEnv {
 public:
  std::vector<ITestCase*> tests;
  std::atomic_int next_test_to_run;
  TestResultStat& stat;
  std::unique_ptr<FixedCountFinishCallback> finished;
  const SessionFactory& sf;
  TestEnv(const std::vector<ITestCase*>& tests, TestResultStat& stat1, SessionFactory& sf1);

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TestEnv);
};
