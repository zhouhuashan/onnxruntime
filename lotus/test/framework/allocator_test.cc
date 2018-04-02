#include "core/framework/allocatormgr.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
TEST(AllocatorTest, CPUAllocatorTest) {
  auto& allocator_manager = AllocatorManager::Instance();

  auto& cpu_arena = allocator_manager.GetArena(CPU);
  EXPECT_EQ(cpu_arena.Info().name, CPU);
  EXPECT_EQ(cpu_arena.Info().id, 0);
  EXPECT_EQ(cpu_arena.Info().type, AllocatorType::kArenaAllocator);

  size_t size = 1024;
  auto bytes = cpu_arena.Alloc(size);
  EXPECT_TRUE(bytes);
  //test the bytes are ok for read/write
  memset(bytes, -1, 1024);

  EXPECT_EQ(*((int*)bytes), -1);
  cpu_arena.Free(bytes);
  //todo: test the used / max api.
}
}  // namespace Test
}  // namespace Lotus
