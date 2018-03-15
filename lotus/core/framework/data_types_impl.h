#ifndef CORE_FRAMEWORK_DATATYPES_IMPL_H
#define CORE_FRAMEWORK_DATATYPES_IMPL_H

#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

namespace Lotus {
    template<typename T>
    static void Delete(void* p)
    {
        delete static_cast<T*>(p);
    }

    template<typename T>
    class RegistreredType : public DataTypeImpl {
    public:
        static MLDataType Type() {
            static RegistreredType<T> s;
            return &s;
        }

        virtual const size_t Size() const override{
            return sizeof(T);
        }

        virtual DeleteFunc GetDeleteFunc() const override {
            return &Delete<T>;
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
    //    static MLDataType Type(
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
    MLDataType DataTypeImpl::GetType<T>()       \
    {                                           \
        return RegistreredType<T>::Type();       \
    }
}

#endif


