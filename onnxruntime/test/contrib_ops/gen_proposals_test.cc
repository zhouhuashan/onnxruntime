// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(GenerateProposalsTest, PositiveTest) {
  OpTester test("GenerateProposals", 1, onnxruntime::kMSDomain);
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
