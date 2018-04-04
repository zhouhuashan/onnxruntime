#include "gtest/gtest.h"
#include "test/test_utils.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

GTEST_API_ int main(int argc, char** argv) {
  Lotus::Test::DefaultInitialize(argc, argv);
  // Register microsoft domain with min/max op_set version as 1/1.
  onnx::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(LotusIR::kMSDomain, 1, 1);
  // Register microsoft domain ops.
  LotusIR::MsOpRegistry::RegisterMsOps();
  return RUN_ALL_TESTS();
}
