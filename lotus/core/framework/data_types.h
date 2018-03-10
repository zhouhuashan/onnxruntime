#ifndef CORE_FRAMEWORK_DATA_TYPES_H
#define CORE_FRAMEWORK_DATA_TYPES_H

#include <string>
#include "core/protobuf/onnx-ml.pb.h"
#include "core/common/common.h"

using namespace onnx;

namespace Lotus
{
    class DataTypeImpl;
    // DataTypeImpl pointer as unique DataTypeImpl identifier.
    typedef const DataTypeImpl* MLDataType;

    class DataTypeImpl {
    public:
        virtual ~DataTypeImpl() {}

        virtual bool IsCompatible(const TypeProto& type_proto) const {
            // TODO: this API will be used to check type in runtime really
            // matches type defined in a model.
            // 1) This should be overriden by sub-classes and have detail check implementation.
            // 2) The reason is checking compatibility is because one runtime type may be
            // able to represent multiple type protos, for example, "float" could match float16, float.
            // 3) After sub-class having the implementation of this function in-place, we should either
            // change the return value from true to false here or make this function as a pure virtual function.
            UNUSED_PARAMETER(type_proto);
            return true;
        }

        virtual bool IsTensorType() const {
            return false;
        }
        
        template<typename T>
        static MLDataType GetTensorType();
    };

    class TensorTypeBase : public DataTypeImpl {
    public:

        virtual bool IsTensorType() const override {
            return true;
        }
    };

    template <typename elemT>
    struct TensorType : public TensorTypeBase {

        static MLDataType Type() {
            static TensorType tensor_type;
            return &tensor_type;
        }

    private:
        TensorType() = default;
    };

    template<typename T>
    class NonTensorType : public DataTypeImpl {
    public:
        static MLDataType Type() {
            static NonTensorType non_tensor_type;
            return &non_tensor_type;
        }

    private:
        NonTensorType() = default;
    };

#define LOTUS_REGISTER_TENSOR_TYPE(ELEM_TYPE)              \
    template<>                                             \
    MLDataType DataTypeImpl::GetTensorType<ELEM_TYPE>()    \
    {                                                      \
        return TensorType<ELEM_TYPE>::Type();              \
    }
}

#endif
