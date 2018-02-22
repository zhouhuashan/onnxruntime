#ifndef CORE_FRAMEWORK_DATA_TYPES_H
#define CORE_FRAMEWORK_DATA_TYPES_H

#include <string>
#include "core/protobuf/onnx-ml.pb.h"

using namespace onnx;

namespace Lotus
{
    struct DataTypeImpl;
    // DataTypeImpl pointer as unique DataTypeImpl identifier.
    typedef const DataTypeImpl* MLDataType;

    struct DataTypeImpl {
        virtual ~DataTypeImpl();

        const TypeProto& ToProto() const;
        
        static MLDataType Tensor_FLOAT;
        static MLDataType Tensor_UINT8;
        static MLDataType Tensor_INT8;
        static MLDataType Tensor_UINT16;
        static MLDataType Tensor_INT16;
        static MLDataType Tensor_INT32;
        static MLDataType Tensor_INT64;
        static MLDataType Tensor_STRING;
        static MLDataType Tensor_BOOL;
        static MLDataType Tensor_FLOAT16;
        static MLDataType Tensor_DOUBLE;
        static MLDataType Tensor_UINT32;
        static MLDataType Tensor_UINT64;
        static MLDataType Tensor_COMPLEX64;
        static MLDataType Tensor_COMPLEX128;

        template<typename T>
        static MLDataType GetType();

    protected:

        DataTypeImpl() {}

        TypeProto m_type_proto;

        std::string m_typeid;
    };
}

#endif
