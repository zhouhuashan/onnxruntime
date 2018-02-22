#include "core/framework/allocator.h"
#include <stdlib.h>
#include <sstream>

namespace Lotus {

    void* CPUAllocator::Alloc(size_t size)
    {
        if (size <= 0)
            return nullptr;
        //todo: we should pin the memory in some case
        void* p = malloc(size);
        return p;
    }

    void CPUAllocator::Free(void* p, size_t size)
    {
        //todo: should replaced with an UNUSED_PARAMETER macro;
        (size);
        //todo: unpin the memory
        free(p);
    }

    size_t CPUAllocator::MinChunkSize()
    {
        LOTUS_NOT_IMPLEMENTED;
    }

    size_t CPUAllocator::MaxChunkSize()
    {
        LOTUS_NOT_IMPLEMENTED;
    }

    AllocatorInfo& CPUAllocator::Info()
    {
        static AllocatorInfo cpuAllocatorInfo(CPU, AllocatorType::DeviceAllocator);
        return cpuAllocatorInfo;
    }
}