#include "core/framework/allocatormgr.h"
#include "../test_utils.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

namespace Lotus {
namespace Test {
TEST(AllocatorTest, CUDAAllocatorTest) {
  int cuda_device_id = 0;
  auto& device_factories = DeviceAllocatorRegistry::Instance().AllRegistrations();
  auto cuda_allocator_creator = device_factories.find(CUDA);
  EXPECT_TRUE(cuda_allocator_creator != device_factories.end());
  auto cuda_arena = CreateAllocator(cuda_allocator_creator->second, cuda_device_id);

  size_t size = 1024;

  EXPECT_EQ(cuda_arena->Info().name, CUDA);
  EXPECT_EQ(cuda_arena->Info().id, cuda_device_id);
  EXPECT_EQ(cuda_arena->Info().mem_type, kMemTypeDefault);
  EXPECT_EQ(cuda_arena->Info().type, AllocatorType::kArenaAllocator);

  //test cuda allocation
  auto cuda_addr = cuda_arena->Alloc(size);
  EXPECT_TRUE(cuda_addr);

  auto pinned_allocator_creator = device_factories.find(CUDA_PINNED);
  EXPECT_TRUE(pinned_allocator_creator != device_factories.end());
  auto pinned_allocator = CreateAllocator(pinned_allocator_creator->second);

  EXPECT_EQ(pinned_allocator->Info().name, CUDA_PINNED);
  EXPECT_EQ(pinned_allocator->Info().id, 0);
  EXPECT_EQ(pinned_allocator->Info().mem_type, kMemTypeCPUOutput);
  EXPECT_EQ(pinned_allocator->Info().type, AllocatorType::kArenaAllocator);

  //test pinned allocation
  auto pinned_addr = pinned_allocator->Alloc(size);
  EXPECT_TRUE(pinned_addr);

  const auto& cpu_arena = TestCPUExecutionProvider()->GetAllocator();
  EXPECT_EQ(cpu_arena->Info().name, CPU);
  EXPECT_EQ(cpu_arena->Info().id, 0);
  EXPECT_EQ(cpu_arena->Info().mem_type, kMemTypeDefault);
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

  cudaMemcpyAsync(pinned_addr, cuda_addr, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  EXPECT_EQ(*((int*)pinned_addr), -1);

  cpu_arena->Free(cpu_addr_a);
  cpu_arena->Free(cpu_addr_b);
  cuda_arena->Free(cuda_addr);
  pinned_allocator->Free(pinned_addr);
}
}  // namespace Test
}  // namespace Lotus
