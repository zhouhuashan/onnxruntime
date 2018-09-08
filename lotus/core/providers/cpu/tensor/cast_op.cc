#include "core/providers/cpu/tensor/cast_op.h"
#include <sstream>
using namespace onnx;
namespace onnxruntime {

const std::vector<MLDataType> castOpTypeConstraints{
    DataTypeImpl::GetTensorType<bool>(),
    DataTypeImpl::GetTensorType<float>(),
    DataTypeImpl::GetTensorType<double>(),
    DataTypeImpl::GetTensorType<uint8_t>(),
    DataTypeImpl::GetTensorType<uint16_t>(),
    DataTypeImpl::GetTensorType<uint32_t>(),
    DataTypeImpl::GetTensorType<uint64_t>(),
    DataTypeImpl::GetTensorType<int8_t>(),
    DataTypeImpl::GetTensorType<int16_t>(),
    DataTypeImpl::GetTensorType<int32_t>(),
    DataTypeImpl::GetTensorType<int64_t>(),
    DataTypeImpl::GetTensorType<MLFloat16>()};

#define ADD_FROM_CAST_OP(in_type)                                                                                                  \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                                                                  \
      Cast,                                                                                                                        \
      6,                                                                                                                           \
      in_type,                                                                                                                     \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", castOpTypeConstraints), \
      Cast<in_type>);                                                                                                              \
                                                                                                                                   \
  template <>                                                                                                                      \
  Status Cast<in_type>::Compute(OpKernelContext* context) const {                                                                  \
    const Tensor* X = context->Input<Tensor>(0);                                                                                   \
    const TensorShape& shape = X->Shape();                                                                                         \
    Tensor* Y = context->Output(0, TensorShape(shape));                                                                            \
                                                                                                                                   \
    switch (to_) {                                                                                                                 \
      case TensorProto_DataType_BOOL:                                                                                              \
        CastData<in_type, bool>(X, Y, shape);                                                                                      \
        break;                                                                                                                     \
      case TensorProto_DataType_INT16:                                                                                             \
        CastData<in_type, int16_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_INT32:                                                                                             \
        CastData<in_type, int32_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_INT64:                                                                                             \
        CastData<in_type, int64_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT8:                                                                                             \
        CastData<in_type, uint8_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT16:                                                                                            \
        CastData<in_type, uint16_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT32:                                                                                            \
        CastData<in_type, uint32_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT64:                                                                                            \
        CastData<in_type, uint64_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_FLOAT:                                                                                             \
        CastData<in_type, float>(X, Y, shape);                                                                                     \
        break;                                                                                                                     \
      case TensorProto_DataType_DOUBLE:                                                                                            \
        CastData<in_type, double>(X, Y, shape);                                                                                    \
        break;                                                                                                                     \
      case TensorProto_DataType_INT8:                                                                                              \
        CastData<in_type, int8_t>(X, Y, shape);                                                                                    \
        break;                                                                                                                     \
      case TensorProto_DataType_FLOAT16:                                                                                           \
        if (std::is_same<in_type, float>::value) {                                                                                 \
          CastData<float, MLFloat16>(X, Y, shape);                                                                                 \
        } else {                                                                                                                   \
          CastFloat16Data<in_type, MLFloat16>(X, Y, shape, Info());                                                                \
        }                                                                                                                          \
        break;                                                                                                                     \
      case TensorProto_DataType_STRING:                                                                                            \
        LOTUS_THROW("Casting to and from strings is not supported yet."); /*break;*/                                               \
      case TensorProto_DataType_UNDEFINED:                                                                                         \
        LOTUS_THROW("Cast op must have 'to' argument of type DataType"); /*break;*/                                                \
      default:                                                                                                                     \
        LOTUS_THROW("Unexpected 'to' argument value: ", to_);                                                                      \
    }                                                                                                                              \
    return Status::OK();                                                                                                           \
  }

ADD_FROM_CAST_OP(uint8_t);
ADD_FROM_CAST_OP(uint16_t);
ADD_FROM_CAST_OP(uint32_t);
ADD_FROM_CAST_OP(uint64_t);
ADD_FROM_CAST_OP(int8_t);
ADD_FROM_CAST_OP(int16_t);
ADD_FROM_CAST_OP(int32_t);
ADD_FROM_CAST_OP(int64_t);
ADD_FROM_CAST_OP(bool);
ADD_FROM_CAST_OP(float);
ADD_FROM_CAST_OP(double);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Cast,
    6,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>()).TypeConstraint("T2", castOpTypeConstraints),
    Cast<MLFloat16>);

template <>
Status Cast<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, TensorShape(shape));
  const auto& op_kernel_info = Info();

  switch (to_) {
    case TensorProto_DataType_BOOL:
      CastFloat16Data<MLFloat16, bool>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_INT16:
      CastFloat16Data<MLFloat16, int16_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_INT32:
      CastFloat16Data<MLFloat16, int32_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_INT64:
      CastFloat16Data<MLFloat16, int64_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_UINT8:
      CastFloat16Data<MLFloat16, uint8_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_UINT16:
      CastFloat16Data<MLFloat16, uint16_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_UINT32:
      CastFloat16Data<MLFloat16, uint32_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_UINT64:
      CastFloat16Data<MLFloat16, uint64_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_FLOAT:
      CastData<MLFloat16, float>(X, Y, shape);
      break;
    case TensorProto_DataType_FLOAT16:
      // no op
      break;
    case TensorProto_DataType_DOUBLE:
      CastFloat16Data<MLFloat16, double>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_INT8:
      CastFloat16Data<MLFloat16, int8_t>(X, Y, shape, op_kernel_info);
      break;
    case TensorProto_DataType_STRING:
      LOTUS_THROW("Casting to and from strings is not supported yet."); /*break;*/
    case TensorProto_DataType_UNDEFINED:
      LOTUS_THROW("Cast op must have 'to' argument of type DataType"); /*break;*/
    default:
      LOTUS_THROW("Unexpected 'to' argument value: ", to_);
  }
  return Status::OK();
}

}  //namespace onnxruntime
