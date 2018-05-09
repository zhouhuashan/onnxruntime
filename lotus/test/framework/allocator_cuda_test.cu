#include "core/framework/allocatormgr.h"
#include "test_utils.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace {

}
namespace Test {
TEST(AllocatorTest, CUDAAllocatorTest) {
  auto& device_factories = DeviceAllocatorRegistry::Instance().AllRegistrations();
  auto cuda_allocator_creator = device_factories.find(CUDA);
  EXPECT_TRUE(cuda_allocator_creator != device_factories.end());
  auto cuda_arena = CreateArena(cuda_allocator_creator->second);

  size_t size = 1024;

  EXPECT_EQ(cuda_arena->Info().name, CUDA);
  EXPECT_EQ(cuda_arena->Info().id, 0);
  EXPECT_EQ(cuda_arena->Info().type, AllocatorType::kArenaAllocator);

  //test cuda allocation
  auto cuda_addr = cuda_arena->Alloc(size);
  EXPECT_TRUE(cuda_addr);

  auto& cpu_arena = TestCPUExecutionProvider()->GetAllocator();
  EXPECT_EQ(cpu_arena->Info().name, CPU);
  EXPECT_EQ(cpu_arena->Info().id, 0);
  EXPECT_EQ(cpu_arena->Info().type, AllocatorType::kArenaAllocator);

  auto cpu_addr_a = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_a);
  auto cpu_addr_b = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_b);
  memset(cpu_addr_a, -1, 1024);

  //test host-device memory copy
  cudaMemcpy(cuda_addr, cpu_addr_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(cpu_addr_b, cuda_addr, size, cudaMemcpyDeviceToHost);
  EXPECT_EQ(*((int*)cpu_addr_b), -1);

  cpu_arena->Free(cpu_addr_a);
  cpu_arena->Free(cpu_addr_b);
  cuda_arena->Free(cuda_addr);
}
}  // namespace Test
}  // namespace Lotus
