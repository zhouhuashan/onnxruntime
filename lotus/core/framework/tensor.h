#ifndef CORE_FRAMEWORK_TENSOR_H
#define CORE_FRAMEWORK_TENSOR_H

#include <string>
#include <vector>

#include "core/protobuf/onnx-ml.pb.h"
#include "core/framework/data_types.h"
#include "core/framework/allocator.h"

using namespace onnx;

namespace Lotus
{
    class TensorShape {

    public:
        TensorShape();

        TensorShape(const std::vector<int64_t>& dims);

        TensorShape(const TensorShape& p_other);

        // Return the dimension specified by <p_idx>.
        int64_t& operator[](int p_idx);

        // Return the dimension specified by <p_idx>.
        const int64_t& operator[](int p_idx) const;

        bool operator==(const TensorShape& p_other) const {
            return m_dims == p_other.m_dims;
        }

        bool operator!=(const TensorShape& p_other) const {
            return !(*this == p_other);
        }

        // Return the total number of elements.
        int64_t Size() const;

        // Return a new TensorShape of the dimensions from dimstart to dimend.
        TensorShape Slice(int p_dimstart, int p_dimend) const;

    private:
        // We use negative numbers for unknown symbolic dimension. Each negative
        // number represents a unique symbolic dimension.
        // InlinedVector<int64_t, 4> dims_;
        std::vector<int64_t> m_dims;
    };

    /*
    We want to keep tensor as simple as possible, it is just a placeholder for a piece of memory, with additional shape information.
    Memory is managered by Executor / Workspace, so Tensor just use it, won't do any allocation / release 
    */
    class Tensor {
        friend class TensorUtil;
    public:
        // Create an empty tensor with float type.
        // empty tensor is a tensor with 1-d shape (0,), and 0 elements.
        Tensor();
        // Create a empty tensor with given type
        Tensor(MLDataType p_type);
        // Create tensor with given type, shape, pre-allocate memory and allocator info. 
        Tensor(MLDataType p_type, const TensorShape& p_shape, void* p_data, const AllocatorInfo& alloc, const ptrdiff_t offset = 0);
        
        //Copy constructure and assign op will just pass the shape and memory reference to another tensor.
        //No deep clone / copy happened.
        Tensor(const Tensor& src);
        Tensor& ShallowCopy(const Tensor& other);

        // Returns the data type.
        MLDataType dtype() const { return m_dtype; }

        // Returns the shape of the tensor.
        const TensorShape& shape() const { return m_shape; }

        // Returns the location of the tensor's memory
        const AllocatorInfo& location() const { return m_alloc_info; }

        template<typename T>
        T* mutable_data() 
        {
            //Type check
            LOTUS_ENFORCE(DataTypeImpl::GetTensorType<T>() == m_dtype, "Tensor type mismatch.");
            return reinterpret_cast<T*>(static_cast<char*>(m_pData) + m_byte_offset);
        }

        template<typename T>
        const T* data() const
        {
            //Type check
            LOTUS_ENFORCE(DataTypeImpl::GetTensorType<T>() == m_dtype, "Tensor type mismatch.");
            return reinterpret_cast<const T*>(static_cast<char*>(m_pData) + m_byte_offset);
        }
        // More API methods.
    private:

        void init(MLDataType p_type, const TensorShape& p_shape, void* p_data, const AllocatorInfo& alloc, const ptrdiff_t bytes_offset);

        void* m_pData;         // Make it shared_ptr<void>?
        TensorShape m_shape;
        MLDataType m_dtype;
        AllocatorInfo m_alloc_info;
        ptrdiff_t m_byte_offset;
    };

}

#endif  // CORE_FRAMEWORK_TENSOR_H
