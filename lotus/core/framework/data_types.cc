#include "core/framework/data_types_impl.h"

namespace Lotus
{
    DataTypeImpl::~DataTypeImpl()
    {
    }

    const TypeProto& DataTypeImpl::ToProto() const
    {
        return m_type_proto;
    }

    template <int elemT>
    MLDataType TensorType<elemT>::Type() {
        static TensorType tensor_type;
        return &tensor_type;
    }

    MLDataType DataTypeImpl::Tensor_FLOAT =
        TensorType<TensorProto_DataType_FLOAT>::Type();
    MLDataType DataTypeImpl::Tensor_UINT8 =
        TensorType<TensorProto_DataType_UINT8>::Type();
    MLDataType DataTypeImpl::Tensor_INT8 = TensorType<TensorProto_DataType_INT8>::Type();
    MLDataType DataTypeImpl::Tensor_UINT16 =
        TensorType<TensorProto_DataType_UINT16>::Type();
    MLDataType DataTypeImpl::Tensor_INT16 =
        TensorType<TensorProto_DataType_INT16>::Type();
    MLDataType DataTypeImpl::Tensor_INT32 =
        TensorType<TensorProto_DataType_INT32>::Type();
    MLDataType DataTypeImpl::Tensor_INT64 =
        TensorType<TensorProto_DataType_INT64>::Type();
    MLDataType DataTypeImpl::Tensor_STRING =
        TensorType<TensorProto_DataType_STRING>::Type();
    MLDataType DataTypeImpl::Tensor_BOOL = TensorType<TensorProto_DataType_BOOL>::Type();
    MLDataType DataTypeImpl::Tensor_FLOAT16 =
        TensorType<TensorProto_DataType_FLOAT16>::Type();
    MLDataType DataTypeImpl::Tensor_DOUBLE =
        TensorType<TensorProto_DataType_DOUBLE>::Type();
    MLDataType DataTypeImpl::Tensor_UINT32 =
        TensorType<TensorProto_DataType_UINT32>::Type();
    MLDataType DataTypeImpl::Tensor_UINT64 =
        TensorType<TensorProto_DataType_UINT64>::Type();
    MLDataType DataTypeImpl::Tensor_COMPLEX64 =
        TensorType<TensorProto_DataType_COMPLEX64>::Type();
    MLDataType DataTypeImpl::Tensor_COMPLEX128 =
        TensorType<TensorProto_DataType_COMPLEX128>::Type();

    LOTUS_REGISTRY_TYPE(int);
    LOTUS_REGISTRY_TYPE(float);
    LOTUS_REGISTRY_TYPE(bool);
    LOTUS_REGISTRY_TYPE(std::string);
    LOTUS_REGISTRY_TYPE(uint8_t);
    LOTUS_REGISTRY_TYPE(uint16_t);
    LOTUS_REGISTRY_TYPE(int16_t);
    LOTUS_REGISTRY_TYPE(int64_t);
    LOTUS_REGISTRY_TYPE(double);
    LOTUS_REGISTRY_TYPE(uint32_t);
    LOTUS_REGISTRY_TYPE(uint64_t);
}
