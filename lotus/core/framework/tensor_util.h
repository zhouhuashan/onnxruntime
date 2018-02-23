#ifndef CORE_FRAMEWORK_TENSOR_UTIL_H
#define CORE_FRAMEWORK_TENSOR_UTIL_H

#include "core/common/common.h"
#include "core/framework/tensor.h"

namespace Lotus {
    class TensorUtil
    {
        typedef std::shared_ptr<Tensor> TensorPtr;
    public:
        static TensorPtr ReshapeTensor(const Tensor& tensor, const TensorShape& new_shape)
        {
            LOTUS_ENFORCE(tensor.shape().Size(), new_shape.Size());
            return std::make_shared<Tensor>(tensor.dtype(), new_shape, tensor.m_pData, tensor.m_alloc_info, tensor.m_byte_offset);
        }
    };
    
}

#endif