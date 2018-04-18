#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "allocator.h"
#include "onnx/onnx_pb.h"

namespace Lotus {
class Tensor;
class MLValuePatternPlanner;
struct MemoryBlock;
namespace Utils {
Common::Status GetTensorFromTensorProto(const onnx::TensorProto& tensor_proto, std::unique_ptr<Tensor>* p_tensor, IAllocator& allocator);
std::vector<int64_t> GetTensorShapeFromTensorProto(const onnx::TensorProto& tensor_proto);
std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const onnx::TensorShapeProto& tensor_shape_proto);

Common::Status TraceTensorAllocFromTensorProto(int mlvalue_index, const onnx::TensorProto& tensor_proto, MLValuePatternPlanner* planner);
Common::Status GetTensorFromTensorProtoWithMemoryPattern(
    const onnx::TensorProto& tensor_proto, 
    const AllocatorInfo& location,
    void* buffer_base,
    std::unique_ptr<Tensor>* p_tensor,
    const MemoryBlock& block);
}  // namespace Utils
}  // namespace Lotus
