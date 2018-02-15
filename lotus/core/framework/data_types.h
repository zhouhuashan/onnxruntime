#ifndef CORE_FRAMEWORK_DATA_TYPES_H
#define CORE_FRAMEWORK_DATA_TYPES_H

#include <string>
#include "core/protobuf/onnx-ml.pb.h"

using namespace onnx;

namespace Lotus
{
    struct DataTypeImpl;
    // DataTypeImpl pointer as unique DataTypeImpl identifier.
    typedef const DataTypeImpl* MlDataType;

    struct DataTypeImpl {
        virtual ~DataTypeImpl();

        const TypeProto& ToProto() const;
        
        static MlDataType Tensor_FLOAT;
        static MlDataType Tensor_UINT8;
        static MlDataType Tensor_INT8;
        static MlDataType Tensor_UINT16;
        static MlDataType Tensor_INT16;
        static MlDataType Tensor_INT32;
        static MlDataType Tensor_INT64;
        static MlDataType Tensor_STRING;
        static MlDataType Tensor_BOOL;
        static MlDataType Tensor_FLOAT16;
        static MlDataType Tensor_DOUBLE;
        static MlDataType Tensor_UINT32;
        static MlDataType Tensor_UINT64;
        static MlDataType Tensor_COMPLEX64;
        static MlDataType Tensor_COMPLEX128;

        template<typename T>
        static MlDataType GetType();

    protected:

        DataTypeImpl() {}

        TypeProto m_type_proto;

        std::string m_typeid;
    };
}

#endif
