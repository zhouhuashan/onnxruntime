#pragma once

#include "TestCaseResult.h"
#include <core/framework/ml_value.h>

std::pair<EXECUTE_RESULT, size_t> compareMLValue(const Lotus::MLValue& real, const Lotus::MLValue& expected);