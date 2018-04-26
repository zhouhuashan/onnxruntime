#include "core/framework/allocatormgr.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace {

}
namespace Test {
TEST(AllocatorTest, CUDAAllocatorTest) {
  auto& allocator_manager = AllocatorManager::Instance();
  size_t size = 1024;

  auto& cuda_arena = allocator_manager.GetArena(CUDA);
  EXPECT_EQ(cuda_arena.Info().name, CUDA);
  EXPECT_EQ(cuda_arena.Info().id, 0);
  EXPECT_EQ(cuda_arena.Info().type, AllocatorType::kArenaAllocator);

  //test cuda allocation
  auto cuda_addr = cuda_arena.Alloc(size);
  EXPECT_TRUE(cuda_addr);

  auto& cpu_arena = allocator_manager.GetArena(CPU);
  EXPECT_EQ(cpu_arena.Info().name, CPU);
  EXPECT_EQ(cpu_arena.Info().id, 0);
  EXPECT_EQ(cpu_arena.Info().type, AllocatorType::kArenaAllocator);

  auto cpu_addr_a = cpu_arena.Alloc(size);
  EXPECT_TRUE(cpu_addr_a);
  auto cpu_addr_b = cpu_arena.Alloc(size);
  EXPECT_TRUE(cpu_addr_b);
  memset(cpu_addr_a, -1, 1024);

  //test host-device memory copy
  cudaMemcpy(cuda_addr, cpu_addr_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(cpu_addr_b, cuda_addr, size, cudaMemcpyDeviceToHost);
  EXPECT_EQ(*((int*)cpu_addr_b), -1);

  cpu_arena.Free(cpu_addr_a);
  cpu_arena.Free(cpu_addr_b);
  cuda_arena.Free(cuda_addr);
}
}  // namespace Test
}  // namespace Lotus
