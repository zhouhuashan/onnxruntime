#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace Test {
TEST(CPUExecutionProviderTest, MetadataTest) {
  CPUExecutionProviderInfo info;
  auto provider = std::make_unique<CPUExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  EXPECT_EQ(provider->GetAllocator(kMemTypeDefault)->Info().name, CPU);
}
}  // namespace Test
}  // namespace onnxruntime
