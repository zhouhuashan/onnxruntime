#ifndef CORE_FRAMEWORK_ML_VALUE_H
#define CORE_FRAMEWORK_ML_VALUE_H

#include <string>

#include "core/framework/data_types.h"

namespace Lotus
{
    struct MLValue
    {
        void* m_pData;       // A pointer to the actual data
        MlDataType m_type;   // A unique id for the static type of the data
    };
}

#endif  // CORE_FRAMEWORK_ML_VALUE_H