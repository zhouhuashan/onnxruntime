#include "core/framework/data_types.h"


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
    MlDataType TensorType<elemT>::Type() {
        static TensorType tensor_type;
        return &tensor_type;
    }

    MlDataType DataTypeImpl::Tensor_FLOAT =
        TensorType<TensorProto_DataType_FLOAT>::Type();
    MlDataType DataTypeImpl::Tensor_UINT8 =
        TensorType<TensorProto_DataType_UINT8>::Type();
    MlDataType DataTypeImpl::Tensor_INT8 = TensorType<TensorProto_DataType_INT8>::Type();
    MlDataType DataTypeImpl::Tensor_UINT16 =
        TensorType<TensorProto_DataType_UINT16>::Type();
    MlDataType DataTypeImpl::Tensor_INT16 =
        TensorType<TensorProto_DataType_INT16>::Type();
    MlDataType DataTypeImpl::Tensor_INT32 =
        TensorType<TensorProto_DataType_INT32>::Type();
    MlDataType DataTypeImpl::Tensor_INT64 =
        TensorType<TensorProto_DataType_INT64>::Type();
    MlDataType DataTypeImpl::Tensor_STRING =
        TensorType<TensorProto_DataType_STRING>::Type();
    MlDataType DataTypeImpl::Tensor_BOOL = TensorType<TensorProto_DataType_BOOL>::Type();
    MlDataType DataTypeImpl::Tensor_FLOAT16 =
        TensorType<TensorProto_DataType_FLOAT16>::Type();
    MlDataType DataTypeImpl::Tensor_DOUBLE =
        TensorType<TensorProto_DataType_DOUBLE>::Type();
    MlDataType DataTypeImpl::Tensor_UINT32 =
        TensorType<TensorProto_DataType_UINT32>::Type();
    MlDataType DataTypeImpl::Tensor_UINT64 =
        TensorType<TensorProto_DataType_UINT64>::Type();
    MlDataType DataTypeImpl::Tensor_COMPLEX64 =
        TensorType<TensorProto_DataType_COMPLEX64>::Type();
    MlDataType DataTypeImpl::Tensor_COMPLEX128 =
        TensorType<TensorProto_DataType_COMPLEX128>::Type();
}