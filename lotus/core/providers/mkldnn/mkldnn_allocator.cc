#include "mkldnn_allocator.h"
#include "core/framework/allocatormgr.h"

namespace Lotus {

const AllocatorInfo& MKLDNNAllocator::Info() const {
  static AllocatorInfo mkl_allocator_info(MKLDNN, AllocatorType::kDeviceAllocator);
  return mkl_allocator_info;
}

const AllocatorInfo& MKLDNNCPUAllocator::Info() const {
  static AllocatorInfo mkl_cpu_allocator_info(MKLDNN_CPU, AllocatorType::kDeviceAllocator);
  return mkl_cpu_allocator_info;
}
}  // namespace Lotus
