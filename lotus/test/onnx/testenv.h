#pragma once
#include <atomic>
#include <vector>
#include "TestCaseInfo.h"
#include "TestResultStat.h"
#include "IFinishCallback.h"
#include <core/framework/inference_session.h>
#include <core/common/common.h>
#include <core/framework/allocatormgr.h>

class TestEnv {
 public:
  const std::vector<TestCaseInfo>& tests;
  std::atomic_int next_test_to_run;
  TestResultStat& stat;
  const Lotus::AllocationPlannerType planner;
  std::unique_ptr<IFinishCallback> finished;
  Lotus::AllocatorManager& allocatorManager;
  TestEnv(const std::vector<TestCaseInfo>& tests1, TestResultStat& stat1, Lotus::AllocationPlannerType planner1);

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TestEnv);
};