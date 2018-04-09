#include "tensorprotoutils.h"
#include "core/graph/tensorutils.h"
#include "tensor.h"

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
    tensor_shape_vec[i] = dims[i].dim_value();
  }

  return tensor_shape_vec;
}

template <typename T>
static Common::Status GetTensorByTypeFromTensorProto(const TensorProto& tensor_proto,
                                                     const TensorShape& tensor_shape,
                                                     size_t tensor_size,
                                                     std::unique_ptr<Tensor>* p_tensor, IAllocator& alloc) {
  // TODO how should the buffer for this tensor be allocated? for now assuming CPU allocator
  size_t size_to_allocate = sizeof(T) * tensor_shape.Size();
  T* p_data = static_cast<T*>(alloc.Alloc(size_to_allocate));
  // std::move(BufferUniquePtr(buffer, BufferDeleter(alloc))),
  LOTUS_RETURN_IF_ERROR(Lotus::Utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  p_tensor->reset(new Tensor(DataTypeImpl::GetType<T>(),
                             tensor_shape,
                             static_cast<void*>(p_data),
                             alloc.Info(),
                             &alloc));

  return Common::Status::OK();
}

Common::Status GetTensorFromTensorProto(const TensorProto& tensor_proto, std::unique_ptr<Tensor>* p_tensor, IAllocator& allocator) {
  vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  if (tensor_shape_vec.empty()) {
    std::ostringstream ostr;
    ostr << "Shape is empty for tensor_proto name: " << tensor_proto.name();
    return Common::Status(Common::LOTUS, Common::FAIL, ostr.str());
  }

  TensorShape tensor_shape{tensor_shape_vec};
  size_t tensor_size = tensor_shape.Size();
  switch (tensor_proto.data_type()) {
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
      return GetTensorByTypeFromTensorProto<float>(tensor_proto, tensor_shape, tensor_size, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
      return GetTensorByTypeFromTensorProto<bool>(tensor_proto, tensor_shape, tensor_size, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
      return GetTensorByTypeFromTensorProto<int32_t>(tensor_proto, tensor_shape, tensor_size, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
      return GetTensorByTypeFromTensorProto<int64_t>(tensor_proto, tensor_shape, tensor_size, p_tensor, allocator);
    case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
      return GetTensorByTypeFromTensorProto<std::string>(tensor_proto, tensor_shape, tensor_size, p_tensor, allocator);
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT, ostr.str());
    }
  }
}
}  // namespace Utils
}  // namespace Lotus
