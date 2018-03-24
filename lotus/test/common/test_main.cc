#include <iostream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

GTEST_API_ int main(int argc, char** argv) {
  std::cout << "Running Lotus/test/common tests" << std::endl;
  testing::InitGoogleMock(&argc, argv);
  // testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
