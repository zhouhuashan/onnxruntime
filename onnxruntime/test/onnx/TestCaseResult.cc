#include "TestCaseResult.h"

void TestCaseResult::SetResult(size_t task_id, EXECUTE_RESULT r) {
  excution_result_[task_id] = r;
}
