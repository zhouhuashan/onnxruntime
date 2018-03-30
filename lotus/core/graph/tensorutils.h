#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "onnx/onnx-ml.pb.h"

namespace Lotus {
namespace Utils {
class TensorUtils {
 public:
#define DEFINE_UNPACK_TENSOR(T, Type, field_name, field_size)                                             \
  static Status UnpackTensor(const onnx::TensorProto& tensor, /*out*/ T* p_data, int64_t expected_size) { \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                \
      return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);                                 \
    }                                                                                                     \
    if (tensor.has_raw_data()) {                                                                          \
      if (tensor.raw_data().size() != ((expected_size) * sizeof(T)))                                      \
        return Status(StatusCategory::LOTUS, StatusCode::FAIL,                                            \
                      "UnpackTensor: the pre-allocated size does not match the raw data size");           \
      UnpackTensorWithRawData(tensor, p_data);                                                            \
      return Status::OK();                                                                                \
    }                                                                                                     \
    if (tensor.field_size() != expected_size)                                                             \
      return Status(StatusCategory::LOTUS, StatusCode::FAIL,                                              \
                    "UnpackTensor: the pre-allocated size does not match the size in proto");             \
    for (auto elem : tensor.field_name()) {                                                               \
      *p_data++ = static_cast<T>(elem);                                                                   \
    }                                                                                                     \
    return Status::OK();                                                                                  \
  }

  DEFINE_UNPACK_TENSOR(float, onnx::TensorProto_DataType_FLOAT, float_data, float_data_size);
  DEFINE_UNPACK_TENSOR(int32_t, onnx::TensorProto_DataType_INT32, int32_data, int32_data_size);
  DEFINE_UNPACK_TENSOR(int64_t, onnx::TensorProto_DataType_INT64, int64_data, int64_data_size);

  static Status UnpackTensor(const onnx::TensorProto& tensor, /*out*/ std::string* p_data, int64_t expected_size);
  static Status UnpackTensor(const onnx::TensorProto& tensor, /*out*/ bool* p_data, int64_t expected_size);

 private:
  static inline bool IsLittleEndianOrder() {
    static int n = 1;
    return (*(char*)&n == 1);
  }

  template <typename T>
  static void UnpackTensorWithRawData(const onnx::TensorProto& tensor, /*out*/ T* p_data) {
    auto& raw_data = tensor.raw_data();
    auto buff = raw_data.c_str();
    size_t type_size = sizeof(T);

    if (IsLittleEndianOrder()) {
      memcpy((void*)p_data, (void*)buff, raw_data.size() * sizeof(char));
    } else {
      for (size_t i = 0; i < raw_data.size(); i += type_size, buff += type_size) {
        T result;
        const char* tempBytes = reinterpret_cast<char*>(&result);
        for (size_t j = 0; j < type_size; ++j) {
          memcpy((void*)&tempBytes[j], (void*)&buff[type_size - 1 - i], sizeof(char));
        }
        p_data[i] = result;
      }
    }
  }
};
}  // namespace Utils
}  // namespace Lotus
