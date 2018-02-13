#ifndef CORE_FRAMEWORK_TENSOR_H
#define CORE_FRAMEWORK_TENSOR_H

#include <string>
#include <vector>

#include "core/protobuf/onnx-ml.pb.h"

using namespace onnx;

namespace Lotus
{
    class TensorShape {

    public:
        TensorShape();

        TensorShape(const TensorShape& p_other);

        // Return the dimension specified by <p_idx>.
        int64_t& operator[](int p_idx);

        // Return the dimension specified by <p_idx>.
        const int64_t& operator[](int p_idx) const;

        bool operator==(const TensorShape& p_other) const;

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


    class Tensor {
    public:
        Tensor();
        Tensor(TensorProto_DataType p_type);
        Tensor(TensorProto_DataType p_type, const TensorShape& p_shape, void* p_pData);

        // Returns the data type.
        TensorProto_DataType dtype() const { return m_dtype; }

        // Returns the shape of the tensor.
        const TensorShape& shape() const { return m_shape; }

        // More API methods.
    private:
        void* m_pData;         // Make it shared_ptr<void>?
        TensorShape m_shape;
        TensorProto_DataType m_dtype;
        std::string m_providerid;
        int64_t m_byte_offset;
    };

}

#endif  // CORE_FRAMEWORK_TENSOR_H