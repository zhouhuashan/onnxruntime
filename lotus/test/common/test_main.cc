#include "gtest/gtest.h"
#include "test/test_utils.h"

GTEST_API_ int main(int argc, char** argv) {
  // we test logging in this library so need to control the default logging manager
  // from within those tests
  const bool create_default_logging_manager = false;

  Lotus::Test::DefaultInitialize(argc, argv, create_default_logging_manager);

  return RUN_ALL_TESTS();
}
