#ifndef CORE_FRAMEWORK_ALLOCATOR_H
#define CORE_FRAMEWORK_ALLOCATOR_H

#include <string>
#include "core/framework/tensor.h"
#include "core/graph/status.h"
#include <map>

using namespace Lotus::Common;

namespace Lotus
{
    // The resource allocator on a physcial device.
    // This allocator will directly allocate resource from system call
    class IResourceAllocator
    {
    public:
        virtual ~IResourceAllocator() {}
        virtual void* Alloc(size_t size) = 0;
        virtual void Free(void* p, size_t size) = 0;
        virtual size_t MinChunkSize() = 0;
        virtual size_t MaxChunkSize() = 0;
    };

    // The interface for arena which manage memory allocations
    class IArenaAllocator
    {
    public:
        virtual IArenaAllocator() {}
        // Alloc call need to be thread safe.
        virtual void* Alloc(size_t size) = 0;
        // Free call need to be thread safe.
        virtual void Free(void* p, size_t size) = 0;
        virtual size_t Used() = 0;
        virtual size_t Max() = 0;
    };

    // Default implementation for Arena
    // Arena will hold a pool of pre-allocate memories and manage their lifecycle.
    // Need an underline IResourceAllocator to allocate memories.
    // The setting like max_chunk_size is init by IDeviceDescriptor from resource allocator
    class ArenaBase : IArenaAllocator
    {
    public:
        ArenaBase(IResourceAllocator resource_allocator);

    private:
        IResourceAllocator m_allocator;
    };

    typedef std::function<bool(Tensor& src, Tensor& dist)> TensorCopier;

    class AllocatorContext
    {
    public:

        static AllocatorContext Instance()
        {
            static AllocatorContext context;
            return context;
        }

        // TODO: registration for customized copy method.

    private:

        AllocatorContext() {}

        std::unordered_map<std::pair<std::string, std::string>, TensorCopier> m_copy_methods;
    };
}
#endif  // CORE_FRAMEWORK_ALLOCATOR_H
