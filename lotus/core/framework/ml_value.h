#ifndef CORE_FRAMEWORK_ML_VALUE_H
#define CORE_FRAMEWORK_ML_VALUE_H

#include <string>

#include "core/framework/data_types.h"

namespace Lotus
{
    struct MLValue
    {
        void* pData;       // A pointer to the actual data
        MLDataType type;   // A unique id for the static type of the data
    };
}

#endif  // CORE_FRAMEWORK_ML_VALUE_H
