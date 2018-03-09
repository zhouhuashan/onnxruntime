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
        std::string name_;
        int id_;
        AllocatorType type_;

    public:
        AllocatorInfo(const std::string& name, AllocatorType type, const int id = 0)
            : name_(name),
            id_(id),
            type_(type)
        {}

        bool operator==(const AllocatorInfo& p_other) const {
            return name_ == p_other.name_
                && id_ == p_other.id_
                && type_ == p_other.type_;
        }
    };

    class IAllocator
    {
    public:
        virtual ~IAllocator() {}
        virtual void* Alloc(size_t size) = 0;
        virtual void Free(void* p) = 0;
        virtual const AllocatorInfo& Info() const = 0;
    };

    // The resource allocator on a physcial device.
    // This allocator will directly allocate resource from system call
    class IDeviceAllocator : public IAllocator
    {
    public:
        virtual ~IDeviceAllocator() {}
        virtual void* Alloc(size_t size) = 0;
        virtual void Free(void* p) = 0;
        virtual size_t MinChunkSize() = 0;
        virtual size_t MaxChunkSize() = 0;
        virtual const AllocatorInfo& Info() const = 0;
    };

    class CPUAllocator : public IDeviceAllocator
    {
    public:
        virtual void* Alloc(size_t size) override;
        virtual void Free(void* p) override;
        virtual size_t MinChunkSize() override;
        virtual size_t MaxChunkSize() override;
        virtual const AllocatorInfo& Info() const override;
    };
}

#endif  // CORE_FRAMEWORK_ALLOCATOR_H
