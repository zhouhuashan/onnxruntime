#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
TEST(CPUExecutionProviderTest, MetadataTest) {
  ExecutionProviderInfo info;
  auto provider = std::make_unique<CPUExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  EXPECT_EQ(provider->GetTempSpaceAllocator().Info().name, CPU);
}
}  // namespace Test
}  // namespace Lotus
