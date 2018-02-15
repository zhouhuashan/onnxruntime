#ifndef CORE_FRAMEWORK_DATATYPES_IMPL_H
#define CORE_FRAMEWORK_DATATYPES_IMPL_H

#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

namespace Lotus {
    template <int elemT>
    struct TensorType : public DataTypeImpl {
        static MlDataType Type();

    private:
        TensorType() {
            m_type_proto.mutable_tensor_type()->set_elem_type(
                (TensorProto_DataType)elemT);
            m_typeid = typeid(Tensor).name();
        }
    };

    template<typename T>
    class RegistreredType : public DataTypeImpl {
    public:
        static MlDataType Type() {
            static RegistreredType<T> s;
            return &s;
        }

    private:
        RegistreredType() {
            // todo: don't know how to handle to proto.
            m_typeid = typeid(T).name();
        }
    };


    //// TODO: enable it after introduced abstract type in proto.
    //template <typename T>
    //struct Abstract : public DataType {
    //    static MlDataType Type(
    //        const std::string& identifier,
    //        const std::string& domain = ONNX_DOMAIN);

    //private:
    //    Abstract(
    //        const std::string& identifier,
    //        const std::string& domain_ = ONNX_DOMAIN) {
    //        type_proto.mutable_abs_type()->set_domain(domain);
    //        type_proto.mutable_abs_type()->set_identifier(identifier);
    //        m_typeid = typeid(T).name();
    //    }
    //};

#define LOTUS_REGISTRY_TYPE(T)                  \
    template<>                                  \
    MlDataType DataTypeImpl::GetType<T>()       \
    {                                           \
        return RegistreredType<T>::Type();       \
    }
}

#endif


