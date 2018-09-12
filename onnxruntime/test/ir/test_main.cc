#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;

  try {
    Lotus::Test::TestEnvironment test_environment{argc, argv};

    // Register Microsoft domain with min/max op_set version as 1/1.
    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(LotusIR::kMSDomain, 1, 1);

    // Register Microsoft domain ops.
    LotusIR::MsOpRegistry::RegisterMsOps();

    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }

  return status;
}
