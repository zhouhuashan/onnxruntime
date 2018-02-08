#ifndef CORE_FRAMEWORK_WORKSPACE_H
#define CORE_FRAMEWORK_WORKSPACE_H

#include <string>
#include "core/framework/tensor.h"

namespace Lotus
{
    // A work space managing all data (inputs/outputs) during graph evaluation.
    // It calls AllocatorManager to get specific allocator for given execution provider,
    // and then delegate all tensor creation and deletion to the allocator.
    class WorkSpace
    {
    public:

        // Get a tensor with specific tensor name for specific execution provider.
        // If copy needed, <*this> work space will call specific execution provider's allocator to do it.
        const Tensor* GetTensor(const std::string& p_tensorName, const std::string& p_executionProviderID) const;

        // Create a tensor with specific tensor name for specific execution provider.
        // It should check that there's no name duplication firstly and then call specific execution 
        // provider's allocator to do the creation.
        Tensor* CreateTensor(const std::string& p_tensorName, const std::string& p_executionProviderID);

        // Release a tensor.
        void ReleaseTensor(Tensor* p_tensor);
    };
}

#endif  // CORE_FRAMEWORK_WORKSPACE_H