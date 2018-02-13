#ifndef CORE_FRAMEWORK_ALLOCATORMGR_H
#define CORE_FRAMEWORK_ALLOCATORMGR_H

#include "core/framework/allocator.h"

namespace Lotus {
    class AllocatorManager
    {
        friend class Initializer;
    public:
        // the allocator manager is a global object for entire process.
        // all the inference engine in the same process will use the same allocator manager.
        static AllocatorManager* Instance()
        {
            static AllocatorManager manager;
            return &manager;
        }

        IArenaAllocator& GetArena(const std::string& name, const int id = 0);

    private:
        // after add allocator, allocator manager will take the ownership.
        Common::Status AddDeviceAllocator(IDeviceAllocator* allocator, const bool create_arena = true);
        Common::Status AddArenaAllocator(IArenaAllocator* allocator);

        static std::string GetAllocatorId(const std::string& name, const int id, const bool isArena);

        Common::Status InitializeAllocators();
    };
}

#endif
