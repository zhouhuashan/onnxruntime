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
class TensorUtils {
 public:
  template <typename T>
  static Status UnpackTensor(const onnx::TensorProto& tensor,
                             /*out*/ T* p_data,
                             int64_t expected_size);

};  // namespace Utils
}  // namespace Utils
}  // namespace onnxruntime
