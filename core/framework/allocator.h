#ifndef CORE_FRAMEWORK_ALLOCATOR_H
#define CORE_FRAMEWORK_ALLOCATOR_H

#include <string>
#include "core\framework\tensor.h"
#include "core\graph\status.h"

using namespace Lotus::Common;

namespace Lotus
{
    // AllocatorMgr 
    // TODO: Register allocator key to allocator class.
    // TODO: Register execution provider key to allocator key.

    class AllocatorManager;
    class IArenaAllocator;
    class IResourceAllocator;

}
#endif  // CORE_FRAMEWORK_ALLOCATOR_H
