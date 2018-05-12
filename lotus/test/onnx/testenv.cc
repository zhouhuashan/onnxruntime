#include "testenv.h"
#include "FixedCountFinishCallback.h"

TestEnv::TestEnv(const std::vector<TestCaseInfo>& tests1, TestResultStat& stat1, Lotus::AllocationPlannerType planner1, const std::string& provider1)
    : next_test_to_run(0), tests(tests1), stat(stat1), planner(planner1), finished(new FixedCountFinishCallback((int)tests1.size())), allocatorManager(Lotus::Test::AllocatorManager::Instance()), provider(provider1) {
}
