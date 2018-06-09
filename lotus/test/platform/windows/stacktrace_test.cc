#if defined(_MSC_VER) && defined(_DEBUG)

#include "core/common/common.h"

#include <iostream>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace Lotus {
namespace Test {

using namespace ::testing;

TEST(StacktraceTests, TestDirectCall) {
  auto result = Lotus::GetStackTrace();
  EXPECT_THAT(result[0], HasSubstr("TestDirectCall"));
}

TEST(StacktraceTests, TestInException) {
  try {
    LOTUS_THROW("Testing");
  } catch (const LotusException& ex) {
    auto msg = ex.what();
    std::cout << msg;

    EXPECT_THAT(msg, HasSubstr("TestInException"));
  }
}

}  // namespace Test
}  // namespace Lotus
#endif
