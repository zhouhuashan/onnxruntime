#pragma once

#include "TestCaseResult.h"
#include <core/framework/ml_value.h>

std::pair<EXECUTE_RESULT, size_t> CompareMLValue(const Lotus::MLValue& real, const Lotus::MLValue& expected, const double abs_error);