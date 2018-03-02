#include "gtest/gtest.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace Lotus
{
    namespace Test
    {
        TEST(CPUExecutionProviderTest, MetadataTest)
        {
            ExecutionProviderInfo info("CPUExecutionProvider", "0.1", NULL);
            auto provider = ExecutionProviderMgr::Instance().GetProvider(info.Name(), info);
            EXPECT_TRUE(provider != nullptr);
            EXPECT_EQ(provider->Name(), info.Name());
            EXPECT_EQ(provider->Version(), info.Version());
            EXPECT_EQ(provider->ID(), "CPUExecutionProvider.0.1");
            EXPECT_EQ(provider->GetAllocator().Info().m_name, CPU);
        }
    }
}