#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "allocator.h"
#include "onnx/onnx-ml.pb.h"

namespace Lotus {
class Tensor;
namespace Utils {
Common::Status GetTensorFromTensorProto(const onnx::TensorProto& tensor_proto, std::unique_ptr<Tensor>* p_tensor, IAllocator& allocator);
std::vector<int64_t> GetTensorShapeFromTensorProto(const onnx::TensorProto& tensor_proto);
std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const onnx::TensorShapeProto& tensor_shape_proto);
}  // namespace Utils
}  // namespace Lotus
