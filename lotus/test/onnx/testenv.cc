#include "testenv.h"
#include "FixedCountFinishCallback.h"

TestEnv::TestEnv(const std::vector<TestCaseInfo>& tests1, const std::vector<std::string>& all_implemented_ops1, TestResultStat& stat1, Lotus::AllocationPlannerType planner1)
    : next_test_to_run(0), tests(tests1), all_implemented_ops(all_implemented_ops1), stat(stat1), planner(planner1), finished(new FixedCountFinishCallback((int)tests1.size())) {}