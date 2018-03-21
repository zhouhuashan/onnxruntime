#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
TEST(CPUExecutionProviderTest, MetadataTest) {
  ExecutionProviderInfo info;
  auto provider = ExecutionProviderMgr::Instance().GetProvider("CPUExecutionProvider", info);
  EXPECT_TRUE(provider != nullptr);
  EXPECT_EQ(provider->GetTempSpaceAllocator().Info().name_, CPU);
}
}  // namespace Test
}  // namespace Lotus