#include "core/providers/cpu/tensor/cast_op.h"

namespace Lotus {

const std::vector<MLDataType> castOpTypeConstraints{
    DataTypeImpl::GetTensorType<bool>(),
    DataTypeImpl::GetTensorType<float>(),
    DataTypeImpl::GetTensorType<double>(),
    DataTypeImpl::GetTensorType<uint8_t>(),
    DataTypeImpl::GetTensorType<uint16_t>(),
    DataTypeImpl::GetTensorType<uint32_t>(),
    DataTypeImpl::GetTensorType<uint64_t>(),
    DataTypeImpl::GetTensorType<int16_t>(),
    DataTypeImpl::GetTensorType<int32_t>(),
    DataTypeImpl::GetTensorType<int64_t>(),
    DataTypeImpl::GetTensorType<MLFloat16>()};

REGISTER_KERNEL(KernelDefBuilder("Cast")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("T2", castOpTypeConstraints),
                Cast<float>);

REGISTER_KERNEL(KernelDefBuilder("Cast")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
                    .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
                Cast<MLFloat16>);

template <>
Status Cast<float>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, TensorShape(shape));

  switch (to_) {
    case TensorProto_DataType_BOOL:
      CastData<float, bool>(X, Y, shape);
      break;
    case TensorProto_DataType_INT16:
      CastData<float, int16_t>(X, Y, shape);
      break;
    case TensorProto_DataType_INT32:
      CastData<float, int32_t>(X, Y, shape);
      break;
    case TensorProto_DataType_INT64:
      CastData<float, int64_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT8:
      CastData<float, uint8_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT16:
      CastData<float, uint16_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT32:
      CastData<float, uint32_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT64:
      CastData<float, uint64_t>(X, Y, shape);
      break;
    case TensorProto_DataType_FLOAT:
      CastData<float, float>(X, Y, shape);
      break;
    case TensorProto_DataType_DOUBLE:
      CastData<float, double>(X, Y, shape);
      break;
    case TensorProto_DataType_INT8:
      LOTUS_THROW("Casting to and from int8_t is not supported yet.");
      //break;
    case TensorProto_DataType_STRING:
      LOTUS_THROW("Casting to and from strings is not supported yet.");
      // break;
    case TensorProto_DataType_FLOAT16:
      CastData<float, MLFloat16>(X, Y, shape);
      break;
    case TensorProto_DataType_UNDEFINED:
      LOTUS_THROW("Cast op must have 'to' argument of type DataType");
      // break;
    default:
      LOTUS_THROW("Unexpected 'to' argument value: ", to_);
  }
  return Status::OK();
}

template <>
Status Cast<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, TensorShape(shape));
  if (to_ != TensorProto_DataType_FLOAT)
    //todo: append the to_ type to string
    return Status(LOTUS, FAIL, "Cast from float16 to unsupported type.");
  CastData<MLFloat16, float>(X, Y, shape);
  return Status::OK();
}

}  //namespace Lotus
