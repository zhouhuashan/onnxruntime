#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"

namespace onnx {
class TensorProto;
class TensorShapeProto;
}  // namespace onnx

namespace Lotus {
class Tensor;
class MLValuePatternPlanner;
namespace Utils {
Common::Status GetTensorFromTensorProto(const onnx::TensorProto& tensor_proto, std::unique_ptr<Tensor>* p_tensor, AllocatorPtr allocator, void* preallocated = nullptr, size_t preallocated_size = 0);
std::vector<int64_t> GetTensorShapeFromTensorProto(const onnx::TensorProto& tensor_proto);
std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const onnx::TensorShapeProto& tensor_shape_proto);

Common::Status TraceTensorAllocFromTensorProto(int mlvalue_index, const onnx::TensorProto& tensor_proto, MLValuePatternPlanner* planner);
}  // namespace Utils
}  // namespace Lotus
