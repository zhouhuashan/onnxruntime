#pragma once

#include <type_traits>
#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"

namespace onnx {
class TensorProto;
}
namespace onnxruntime {
namespace Utils {
//How much memory it will need for putting the content of this tensor into a plain array
//string/complex64/complex128 tensors are not supported.
//The output value could be zero or -1.
common::Status GetSizeInBytesFromTensorProto(const onnx::TensorProto& tensor_proto, size_t* out);
class TensorUtils {
 public:
  template <typename T>
  static Status UnpackTensor(const onnx::TensorProto& tensor,
                             /*out*/ T* p_data,
                             int64_t expected_size);

};  // namespace Utils
}  // namespace Utils
}  // namespace onnxruntime
