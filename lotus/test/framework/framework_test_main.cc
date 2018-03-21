#include <iostream>
#include "core/framework/init.h"
#include "gtest/gtest.h"

GTEST_API_ int main(int argc, char** argv) {
  std::cout << "Running main() from framework_test_main.cc" << std::endl;
  testing::InitGoogleTest(&argc, argv);
  Lotus::Initializer::EnsureInitialized(&argc, &argv);
  return RUN_ALL_TESTS();
}