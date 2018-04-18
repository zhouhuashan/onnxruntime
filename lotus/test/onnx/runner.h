#pragma once
#include <vector>
#include <string>
#include "testenv.h"

struct TestCaseResult {
	std::vector<EXECUTE_RESULT> excution_result;
	//only valid for single node tests;
	std::string node_name;
};
void RunSingleTestCase(TestEnv& env, size_t test_index, std::function<void(TestCaseResult& result)> on_finished);

#ifdef _WIN32
extern void ParallelRunTests(TestEnv& env, int p_models, std::vector<TestCaseResult>& results);
#endif