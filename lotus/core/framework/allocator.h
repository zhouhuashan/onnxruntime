#ifndef CORE_FRAMEWORK_ALLOCATOR_H
#define CORE_FRAMEWORK_ALLOCATOR_H

#include <string>
#include "core/common/status.h"
#include <map>
#include "core/common/common.h"
#include "core/common/exceptions.h"

using namespace Lotus::Common;

namespace Lotus
{
#define CPU "CPU"

    enum AllocatorType {
        DeviceAllocator = 0,
        ArenaAllocator = 1
    };

    struct AllocatorInfo
    {
        // use string for name, so we could have customzied allocator in execution provider.
        std::string m_name;
        int m_allocator_id;
        AllocatorType m_type;

    public:
        AllocatorInfo(const std::string& name, AllocatorType type, const int id = 0)
            : m_name(name),
            m_allocator_id(id),
            m_type(type)
        {}
    };

    // The resource allocator on a physcial device.
    // This allocator will directly allocate resource from system call
    class IDeviceAllocator
    {
    public:
        virtual ~IDeviceAllocator() {}
        virtual void* Alloc(size_t size) = 0;
        virtual void Free(void* p, size_t size) = 0;
        virtual size_t MinChunkSize() = 0;
        virtual size_t MaxChunkSize() = 0;
        virtual AllocatorInfo& Info() = 0;
    };

    class CPUAllocator : public IDeviceAllocator
    {
    public:
        virtual void* Alloc(size_t size) override;
        virtual void Free(void* p, size_t size) override;
        virtual size_t MinChunkSize() override;
        virtual size_t MaxChunkSize() override;
        virtual AllocatorInfo& Info() override;
    };

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
        virtual size_t Used() = 0;
        virtual size_t Max() = 0;
        virtual AllocatorInfo& Info() = 0;
    };

    // Dummy Arena which just call underline device allocator directly.
    class DummyArena : public IArenaAllocator
    {
    public:
        DummyArena(IDeviceAllocator* resource_allocator) 
            : m_allocator(resource_allocator), 
            m_info(resource_allocator->Info().m_name, AllocatorType::ArenaAllocator, resource_allocator->Info().m_allocator_id)
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

        virtual size_t Used() override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual size_t Max() override
        {
            LOTUS_NOT_IMPLEMENTED;
        }

        virtual AllocatorInfo& Info() override
        {
            return m_info;
        }

    private:
        IDeviceAllocator* m_allocator;
        AllocatorInfo m_info;
    };

    
}
#endif  // CORE_FRAMEWORK_ALLOCATOR_H
