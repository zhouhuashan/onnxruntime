#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/ml_value.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
class Tensor;
namespace Utils {
common::Status GetTensorFromTensorProto(const onnx::TensorProto& tensor_proto, std::unique_ptr<Tensor>* p_tensor, AllocatorPtr allocator, void* preallocated = nullptr, size_t preallocated_size = 0);
std::vector<int64_t> GetTensorShapeFromTensorProto(const onnx::TensorProto& tensor_proto);
std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const onnx::TensorShapeProto& tensor_shape_proto);
common::Status TensorProtoToMLValue(const onnx::TensorProto& input, AllocatorPtr allocator, void* preallocated,
                                    size_t preallocated_size, MLValue& value);
}  // namespace Utils
}  // namespace onnxruntime
