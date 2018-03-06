#ifndef CORE_FRAMEWORK_ARENA_H
#define CORE_FRAMEWORK_ARENA_H

#include <string>
#include "core/framework/allocator.h"

using namespace Lotus::Common;

namespace Lotus
{
    // The interface for arena which manage memory allocations
    // Arena will hold a pool of pre-allocate memories and manage their lifecycle.
    // Need an underline IResourceAllocator to allocate memories.
    // The setting like max_chunk_size is init by IDeviceDescriptor from resource allocator
    class IArenaAllocator
    {
    public:
        virtual ~IArenaAllocator() {}
        // Alloc call need to be thread safe.
        virtual void* Alloc(size_t size) = 0;
        // Free call need to be thread safe.
        virtual void Free(void* p, size_t size) = 0;
        virtual size_t Used() const = 0;
        virtual size_t Max() const = 0;
        virtual const AllocatorInfo& Info() const = 0;
        // allocate host pinned memory? 
    };

    // Dummy Arena which just call underline device allocator directly.
    class DummyArena : public IArenaAllocator
    {
    public:
        DummyArena(IDeviceAllocator* resource_allocator) 
            : m_allocator(resource_allocator), 
            m_info(resource_allocator->Info().name_, AllocatorType::ArenaAllocator, resource_allocator->Info().id_)
        {
        }

        virtual ~DummyArena() {}

        virtual void* Alloc(size_t size) override
        {
            if (size == 0)
                return nullptr;
            return m_allocator->Alloc(size);
        }

        virtual void Free(void* p, size_t size) override
        {
            m_allocator->Free(p, size);
        }

        virtual size_t Used() const override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual size_t Max() const override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual const AllocatorInfo& Info() const override
        {
            return m_info;
        }

    private:
        IDeviceAllocator* m_allocator;
        AllocatorInfo m_info;
    };
}

#endif  // CORE_FRAMEWORK_ALLOCATOR_H
