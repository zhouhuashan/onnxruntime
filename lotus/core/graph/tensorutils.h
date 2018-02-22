#ifndef LOTUSIR_CORE_GRAPH_TENSORUTILS_H
#define LOTUSIR_CORE_GRAPH_TENSORUTILS_H

#include <vector>

#include "core/protobuf/onnx-ml.pb.h"
#include "core/common/common.h"
#include "core/common/status.h"

namespace Lotus
{
    namespace Utils
    {
        class TensorUtils
        {
        public:
#define DEFINE_UNPACK_TENSOR(T, Type, fieldName, fieldSize)                                                                    \
    static Status UnpackTensor(const onnx::TensorProto& p_tensor, /*out*/T* p_data, int64_t p_expected_size)                   \
    {                                                                                                                          \
        if (Type != p_tensor.data_type()                                                                                       \
            || nullptr == p_data)                                                                                              \
        {                                                                                                                      \
            return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);                                                \
        }                                                                                                                      \
        if (p_tensor.has_raw_data())                                                                                           \
        {                                                                                                                      \
            if (p_tensor.raw_data().size() != (p_expected_size) * sizeof(T))                                                   \
                return Status(StatusCategory::LOTUS, StatusCode::FAIL,                                                         \
                                               "UnpackTensor: the pre-allocate size does not match the raw data size");        \
            UnpackTensorWithRawData(p_tensor, p_data);                                                                         \
            return Status::OK();                                                                                               \
        }                                                                                                                      \
        if (p_tensor.fieldSize() != p_expected_size)                                                                           \
            return Status(StatusCategory::LOTUS, StatusCode::FAIL,                                                             \
                                            "UnpackTensor: the pre-allocate size does not match the size in proto");           \
        for (auto elem : p_tensor.fieldName())                                                                                 \
        {                                                                                                                      \
            *p_data++ = static_cast<T>(elem);                                                                                  \
        }                                                                                                                      \
        return Status::OK();                                                                                                   \
    }

            DEFINE_UNPACK_TENSOR(float, onnx::TensorProto_DataType_FLOAT, float_data, float_data_size);
            DEFINE_UNPACK_TENSOR(int32_t, onnx::TensorProto_DataType_INT32, int32_data, int32_data_size);
            DEFINE_UNPACK_TENSOR(int64_t, onnx::TensorProto_DataType_INT64, int64_data, int64_data_size);

            static Status UnpackTensor(const onnx::TensorProto& p_tensor, /*out*/std::string* p_data, int64_t p_expected_size);
            static Status UnpackTensor(const onnx::TensorProto& p_tensor, /*out*/bool* p_data, int64_t p_expected_size);

        private:

            static inline bool IsLittleEndianOrder()
            {
                static int n = 1;
                return (*(char*)&n == 1);
            }

            template <typename T>
            static void UnpackTensorWithRawData(const onnx::TensorProto& p_tensor, /*out*/T* p_data)
            {
                auto& raw_data = p_tensor.raw_data();
                auto buff = raw_data.c_str();
                size_t typeSize = sizeof(T);

                if (IsLittleEndianOrder())
                {
                    memcpy((void*)p_data, (void*)buff, raw_data.size() * sizeof(char));
                }
                else
                {
                    for (size_t i = 0; i < raw_data.size(); i += typeSize, buff += typeSize)
                    {
                        T result;
                        const char* tempBytes = reinterpret_cast<char*>(&result);
                        for (size_t j = 0; j < typeSize; ++j)
                        {
                            memcpy((void*)&tempBytes[j], (void*)&buff[typeSize - 1 - i], sizeof(char));
                        }
                        p_data[i] = result;
                    }
                }

            }
        };
    }
}

#endif
