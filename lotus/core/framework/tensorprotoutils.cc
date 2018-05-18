#include "tensorprotoutils.h"
#include "core/graph/tensorutils.h"
#include "tensor.h"
#include "core/framework/ml_value_patterns_planner.h"

namespace Lotus {
namespace Utils {
std::vector<int64_t> GetTensorShapeFromTensorProto(const onnx::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<int64_t> tensor_shape_vec(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return tensor_shape_vec;
}

std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const onnx::TensorShapeProto& tensor_shape_proto) {
  const auto& dims = tensor_shape_proto.dim();
  std::vector<int64_t> tensor_shape_vec(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i].has_dim_param()
                              ? -1 /* symbolic dimensions are represented as -1 in Lotus*/
                              : dims[i].dim_value();
  }
  return tensor_shape_vec;
}

template <typename T>
static Common::Status GetTensorByTypeFromTensorProto(const TensorProto& tensor_proto,
                                                     const TensorShape& tensor_shape,
                                                     std::unique_ptr<Tensor>* p_tensor,
                                                     AllocatorPtr alloc) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  int64_t size_to_allocate = static_cast<int64_t>(sizeof(T)) * tensor_size;
  T* p_data = static_cast<T*>(alloc->Alloc(gsl::narrow_cast<size_t>(size_to_allocate)));
  LOTUS_RETURN_IF_ERROR(Lotus::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  p_tensor->reset(new Tensor(DataTypeImpl::GetType<T>(),
                             tensor_shape,
                             static_cast<void*>(p_data),
                             alloc->Info(),
                             alloc));

  return Common::Status::OK();
}

Common::Status GetTensorFromTensorProto(const TensorProto& tensor_proto,
                                        std::unique_ptr<Tensor>* p_tensor,
                                        AllocatorPtr allocator) {
  vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);

  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).

  TensorShape tensor_shape{tensor_shape_vec};

  switch (tensor_proto.data_type()) {
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
      return GetTensorByTypeFromTensorProto<float>(tensor_proto, tensor_shape, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
      return GetTensorByTypeFromTensorProto<double>(tensor_proto, tensor_shape, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
      return GetTensorByTypeFromTensorProto<bool>(tensor_proto, tensor_shape, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
      return GetTensorByTypeFromTensorProto<int32_t>(tensor_proto, tensor_shape, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
      return GetTensorByTypeFromTensorProto<int64_t>(tensor_proto, tensor_shape, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
      return GetTensorByTypeFromTensorProto<std::string>(tensor_proto, tensor_shape, p_tensor, allocator);
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT, ostr.str());
    }
  }
}

template <typename T>
static Common::Status LoadTensorByTypeFromTensorProto(const TensorProto& tensor_proto,
                                                      const TensorShape& tensor_shape,
                                                      size_t tensor_size,
                                                      size_t block_size,
                                                      std::unique_ptr<Tensor>* p_tensor,
                                                      void* buffer,
                                                      const AllocatorInfo& location) {
  T* p_data = static_cast<T*>(buffer);
  size_t data_size = sizeof(T) * tensor_shape.Size();
  if (block_size != data_size)
    return Status(LOTUS, FAIL, "The buffer planner is not consistent with tensor buffer size");
  // std::move(BufferUniquePtr(buffer, BufferDeleter(alloc))),
  LOTUS_RETURN_IF_ERROR(Lotus::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  p_tensor->reset(new Tensor(DataTypeImpl::GetType<T>(),
                             tensor_shape,
                             static_cast<void*>(p_data),
                             location));

  return Common::Status::OK();
}

Common::Status GetTensorFromTensorProtoWithMemoryPattern(
    const onnx::TensorProto& tensor_proto,
    const AllocatorInfo& location,
    void* buffer_base,
    std::unique_ptr<Tensor>* p_tensor,
    const MemoryBlock& block) {
  vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  if (tensor_shape_vec.empty()) {
    std::ostringstream ostr;
    ostr << "Shape is empty for tensor_proto name: " << tensor_proto.name();
    return Common::Status(Common::LOTUS, Common::FAIL, ostr.str());
  }

  TensorShape tensor_shape{tensor_shape_vec};
  size_t tensor_size = tensor_shape.Size();
  void* buffer = static_cast<void*>(static_cast<char*>(buffer_base) + block.offset_);

  switch (tensor_proto.data_type()) {
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
      return LoadTensorByTypeFromTensorProto<float>(tensor_proto, tensor_shape, tensor_size, block.size_, p_tensor, buffer, location);
    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
      return LoadTensorByTypeFromTensorProto<bool>(tensor_proto, tensor_shape, tensor_size, block.size_, p_tensor, buffer, location);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
      return LoadTensorByTypeFromTensorProto<int32_t>(tensor_proto, tensor_shape, tensor_size, block.size_, p_tensor, buffer, location);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
      return LoadTensorByTypeFromTensorProto<int64_t>(tensor_proto, tensor_shape, tensor_size, block.size_, p_tensor, buffer, location);
    case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
      return LoadTensorByTypeFromTensorProto<std::string>(tensor_proto, tensor_shape, tensor_size, block.size_, p_tensor, buffer, location);
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT, ostr.str());
    }
  }
}

template <typename T>
size_t GetTensorSize(const TensorShape& tensor_shape) {
  return sizeof(T) * tensor_shape.Size();
}

Common::Status TraceTensorAllocFromTensorProto(int mlvalue_index, const onnx::TensorProto& tensor_proto, MLValuePatternPlanner* planner) {
  if (!planner)
    return Status(LOTUS, INVALID_ARGUMENT);

  vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  TensorShape tensor_shape{tensor_shape_vec};
  int64_t size = 0;
  switch (tensor_proto.data_type()) {
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: {
      size = GetTensorSize<float>(tensor_shape);
      break;
    }
    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL: {
      size = GetTensorSize<bool>(tensor_shape);
      break;
    }
    case onnx::TensorProto_DataType::TensorProto_DataType_INT32: {
      size = GetTensorSize<int32_t>(tensor_shape);
      break;
    }
    case onnx::TensorProto_DataType::TensorProto_DataType_INT64: {
      size = GetTensorSize<int64_t>(tensor_shape);
      break;
    }
    case onnx::TensorProto_DataType::TensorProto_DataType_STRING: {
      //string tensor size is not predictable, don't plan on this.
      LOGS_DEFAULT(WARNING) << "Can't plan on string tensors.";
      return Status::OK();
    }
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT, ostr.str());
    }
  }

  return planner->TraceAllocation(mlvalue_index, size);
}
}  // namespace Utils
}  // namespace Lotus
