#include "tensorprotoutils.h"

#include <memory>
#include "core/inc/op_kernel_author.h"
#include "core/common/logging/logging.h"
#include "core/graph/tensorutils.h"
#include "tensor.h"
#include "core/framework/ml_value_patterns_planner.h"
using namespace onnx;
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
Common::Status GetTensorByTypeFromTensorProto(const TensorProto& tensor_proto,
                                              const TensorShape& tensor_shape,
                                              std::unique_ptr<Tensor>* p_tensor,
                                              AllocatorPtr alloc,
                                              void* preallocated,
                                              size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  size_t size_to_allocate = sizeof(T) * gsl::narrow_cast<size_t>(tensor_size);

  if (preallocated && preallocated_size != size_to_allocate)
    return Status(LOTUS, FAIL, "The buffer planner is not consistent with tensor buffer size");

  T* p_data = static_cast<T*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  LOTUS_RETURN_IF_ERROR(Lotus::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  return Common::Status::OK();
}

template <>
Common::Status GetTensorByTypeFromTensorProto<std::string>(const TensorProto& tensor_proto,
                                                           const TensorShape& tensor_shape,
                                                           std::unique_ptr<Tensor>* p_tensor,
                                                           AllocatorPtr alloc,
                                                           void* preallocated,
                                                           size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  size_t size_to_allocate = sizeof(std::string) * gsl::narrow_cast<size_t>(tensor_size);

  if (preallocated && preallocated_size != size_to_allocate)
    return Status(LOTUS, FAIL, "The buffer planner is not consistent with tensor buffer size");

  std::string* p_data = static_cast<std::string*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<std::string>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  /*
  In the case of string tensors, the strings need to be constructed in the pre-allocated memory (placement
  new) before calling Unpack (which copies the strings from the proto). Placement new happens inside the
  Tensor's constructor. Hence the order of invocation of Tensor construction and Unpack needs to be reversed
  in comparison to other types. This has the disadvantage of alloc/deallocing a Tensor if Unpack fails;
  however restricting it to string types only alleviates this concern for other types at least. Hence the template
  specialization for string.
  */
  LOTUS_RETURN_IF_ERROR(Lotus::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));

  return Common::Status::OK();
}

template <>
Common::Status GetTensorByTypeFromTensorProto<MLFloat16>(const TensorProto& tensor_proto,
                                                         const TensorShape& tensor_shape,
                                                         std::unique_ptr<Tensor>* p_tensor,
                                                         AllocatorPtr alloc,
                                                         void* preallocated,
                                                         size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  static_assert(sizeof(MLFloat16) == sizeof(uint16_t), "MLFloat16 must has 16 bit size");
  size_t size_to_allocate = sizeof(MLFloat16) * gsl::narrow_cast<size_t>(tensor_size);

  if (preallocated && preallocated_size != size_to_allocate)
    return Status(LOTUS, FAIL, "The buffer planner is not consistent with tensor buffer size");

  uint16_t* p_data = static_cast<uint16_t*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  LOTUS_RETURN_IF_ERROR(Lotus::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<MLFloat16>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  return Common::Status::OK();
}

Common::Status GetTensorFromTensorProto(const TensorProto& tensor_proto,
                                        std::unique_ptr<Tensor>* p_tensor,
                                        AllocatorPtr allocator,
                                        void* preallocated,
                                        size_t preallocated_size) {
  vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);

  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).

  TensorShape tensor_shape{tensor_shape_vec};

  switch (tensor_proto.data_type()) {
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
      return GetTensorByTypeFromTensorProto<float>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);
    case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
      return GetTensorByTypeFromTensorProto<double>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);
    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
      return GetTensorByTypeFromTensorProto<bool>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
      return GetTensorByTypeFromTensorProto<int32_t>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
      return GetTensorByTypeFromTensorProto<int64_t>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);
    case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
      return GetTensorByTypeFromTensorProto<std::string>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
      return GetTensorByTypeFromTensorProto<MLFloat16>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);
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
    case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE: {
      size = GetTensorSize<double>(tensor_shape);
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
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16: {
      size = GetTensorSize<MLFloat16>(tensor_shape);
      break;
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
