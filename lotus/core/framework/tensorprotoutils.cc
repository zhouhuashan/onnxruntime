#include "tensorprotoutils.h"

#include <memory>
#include "core/graph/onnx_protobuf.h"
#include "core/inc/op_kernel_author.h"
#include "core/common/logging/logging.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorutils.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value_patterns_planner.h"

using namespace onnx;
using namespace ::onnxruntime::common;

namespace onnxruntime {
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
                              ? -1 /* symbolic dimensions are represented as -1 in onnxruntime*/
                              : dims[i].dim_value();
  }
  return tensor_shape_vec;
}

template <typename T>
common::Status GetTensorByTypeFromTensorProto(const TensorProto& tensor_proto,
                                              const TensorShape& tensor_shape,
                                              std::unique_ptr<Tensor>* p_tensor,
                                              AllocatorPtr alloc,
                                              void* preallocated,
                                              size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  //tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
  if (tensor_size < 0) {
    return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid shape ", tensor_shape);
  }
  size_t size_to_allocate = sizeof(T) * gsl::narrow<size_t>(tensor_size);

  if (preallocated && preallocated_size != Align256(size_to_allocate))
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "The buffer planner is not consistent with tensor buffer size, expected ", size_to_allocate, ", got ", preallocated_size);
  //TODO(@chasun): size_to_allocate could be zero. We shouldn't pass zero to alloc->Alloc()
  T* p_data = static_cast<T*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  LOTUS_RETURN_IF_ERROR(::onnxruntime::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  return common::Status::OK();
}

template <>
common::Status GetTensorByTypeFromTensorProto<std::string>(const TensorProto& tensor_proto,
                                                           const TensorShape& tensor_shape,
                                                           std::unique_ptr<Tensor>* p_tensor,
                                                           AllocatorPtr alloc,
                                                           void* preallocated,
                                                           size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(common::LOTUS, common::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  size_t size_to_allocate = sizeof(std::string) * gsl::narrow_cast<size_t>(tensor_size);

  if (preallocated && preallocated_size != Align256(size_to_allocate))
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
  LOTUS_RETURN_IF_ERROR(::onnxruntime::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));

  return common::Status::OK();
}

template <>
common::Status GetTensorByTypeFromTensorProto<MLFloat16>(const TensorProto& tensor_proto,
                                                         const TensorShape& tensor_shape,
                                                         std::unique_ptr<Tensor>* p_tensor,
                                                         AllocatorPtr alloc,
                                                         void* preallocated,
                                                         size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(common::LOTUS, common::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  static_assert(sizeof(MLFloat16) == sizeof(uint16_t), "MLFloat16 must has 16 bit size");
  size_t size_to_allocate = sizeof(MLFloat16) * gsl::narrow_cast<size_t>(tensor_size);

  if (preallocated && preallocated_size != Align256(size_to_allocate))
    return Status(LOTUS, FAIL, "The buffer planner is not consistent with tensor buffer size");

  MLFloat16* p_data = static_cast<MLFloat16*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  LOTUS_RETURN_IF_ERROR(::onnxruntime::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<MLFloat16>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  return common::Status::OK();
}

#define LOTUS_CASE_PROTO(X, Y)                               \
  case onnx::TensorProto_DataType::TensorProto_DataType_##X: \
    return GetTensorByTypeFromTensorProto<Y>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);

common::Status GetTensorFromTensorProto(const TensorProto& tensor_proto,
                                        std::unique_ptr<Tensor>* p_tensor,
                                        AllocatorPtr allocator,
                                        void* preallocated,
                                        size_t preallocated_size) {
  std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
  TensorShape tensor_shape{tensor_shape_vec};
  switch (tensor_proto.data_type()) {
    LOTUS_CASE_PROTO(FLOAT, float);
    LOTUS_CASE_PROTO(DOUBLE, double);
    LOTUS_CASE_PROTO(BOOL, bool);
    LOTUS_CASE_PROTO(INT8, int8_t);
    LOTUS_CASE_PROTO(INT16, int16_t);
    LOTUS_CASE_PROTO(INT32, int32_t);
    LOTUS_CASE_PROTO(INT64, int64_t);
    LOTUS_CASE_PROTO(UINT8, uint8_t);
    LOTUS_CASE_PROTO(UINT16, uint16_t);
    LOTUS_CASE_PROTO(UINT32, uint32_t);
    LOTUS_CASE_PROTO(UINT64, uint64_t);
    LOTUS_CASE_PROTO(STRING, std::string);
    LOTUS_CASE_PROTO(FLOAT16, MLFloat16);
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return common::Status(common::LOTUS, common::INVALID_ARGUMENT, ostr.str());
    }
  }
}

template <typename T>
size_t GetTensorSize(const TensorShape& tensor_shape) {
  return sizeof(T) * tensor_shape.Size();
}

#define LOTUS_CASE_PROTO_TRACE(X, Y)                         \
  case onnx::TensorProto_DataType::TensorProto_DataType_##X: \
    size = GetTensorSize<Y>(tensor_shape);                   \
    break;

common::Status TraceTensorAllocFromTensorProto(int mlvalue_index, const onnx::TensorProto& tensor_proto, MLValuePatternPlanner* planner) {
  if (!planner)
    return Status(LOTUS, INVALID_ARGUMENT);

  std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  TensorShape tensor_shape{tensor_shape_vec};
  int64_t size = 0;
  switch (tensor_proto.data_type()) {
    LOTUS_CASE_PROTO_TRACE(FLOAT, float);
    LOTUS_CASE_PROTO_TRACE(DOUBLE, double);
    LOTUS_CASE_PROTO_TRACE(BOOL, bool);
    LOTUS_CASE_PROTO_TRACE(INT8, int8_t);
    LOTUS_CASE_PROTO_TRACE(INT16, int16_t);
    LOTUS_CASE_PROTO_TRACE(INT32, int32_t);
    LOTUS_CASE_PROTO_TRACE(INT64, int64_t);
    LOTUS_CASE_PROTO_TRACE(UINT8, uint8_t);
    LOTUS_CASE_PROTO_TRACE(UINT16, uint16_t);
    LOTUS_CASE_PROTO_TRACE(UINT32, uint32_t);
    LOTUS_CASE_PROTO_TRACE(UINT64, uint64_t);
    LOTUS_CASE_PROTO_TRACE(FLOAT16, MLFloat16);
    case onnx::TensorProto_DataType::TensorProto_DataType_STRING: {
      //string tensor size is not predictable, don't plan on this.
      LOGS_DEFAULT(WARNING) << "Can't plan on string tensors.";
      return Status::OK();
    }
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return common::Status(common::LOTUS, common::INVALID_ARGUMENT, ostr.str());
    }
  }

  return planner->TraceAllocation(mlvalue_index, Align256(size));
}
}  // namespace Utils
}  // namespace onnxruntime
