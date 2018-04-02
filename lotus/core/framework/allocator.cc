#include "core/framework/allocator.h"
#include <stdlib.h>
#include <sstream>

namespace Lotus {

void* CPUAllocator::Alloc(size_t size) {
  if (size <= 0)
    return nullptr;
  //todo: we should pin the memory in some case
  void* p = malloc(size);
  return p;
}

void CPUAllocator::Free(void* p) {
  //todo: unpin the memory
  free(p);
}

size_t CPUAllocator::MinChunkSize() {
  LOTUS_NOT_IMPLEMENTED;
}

size_t CPUAllocator::MaxChunkSize() {
  LOTUS_NOT_IMPLEMENTED;
}

const AllocatorInfo& CPUAllocator::Info() const {
  static AllocatorInfo cpuAllocatorInfo(CPU, AllocatorType::kDeviceAllocator);
  return cpuAllocatorInfo;
}
}  // namespace Lotus
