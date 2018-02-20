#include "core/framework/op_kernel.h"

namespace Lotus {

#define DEFINE_GET_ATTR(T, type)                                                  \
  template <>                                                                     \
  Status OpKernelInfo::GetAttr<T>(                                                \
      const std::string& name, T* value) const {                                  \
    const LotusIR::Node& op_def = OpDef();                                        \
    const LotusIR::NodeAttributes attributes = op_def.GetAttributes();            \
    auto it = attributes.find(name);                                              \
    if (it != attributes.end()) {                                                 \
        const AttributeProto attr = it->second;                                   \
        if (!attr.has_##type()) {                                                 \
            return Status(LOTUS, FAIL, "Attibute name and type don't match");     \
        }                                                                         \
        else{                                                                     \
            *value = static_cast<T>(attr.type());                                 \
            return Status::OK();                                                  \
        }                                                                         \
    }                                                                             \
    return Status(LOTUS, FAIL, "No attribute with this name is defined.");        \
  }                                                                                 

#define DEFINE_GET_ATTRS(T, list)                                                 \
  template <>                                                                     \
  Status OpKernelInfo::GetAttr<T>(                                                \
      const std::string& name, std::vector<T>& values) const {                    \
    const LotusIR::Node& op_def = OpDef();                                        \
    const LotusIR::NodeAttributes attributes = op_def.GetAttributes();            \
    auto it = attributes.find(name);                                              \
    if (it != attributes.end()) {                                                 \
        const AttributeProto attr = it->second;                                   \
        for(int i=0; i<attr.list##_size(); ++i){                                  \
            values.push_back(static_cast<T>(attr.list(i)));                       \
        }                                                                         \
       return Status::OK();                                                       \
    }                                                                             \
    return Status(LOTUS, FAIL, "No attribute with this name is defined.");        \
  }

  DEFINE_GET_ATTR(float, f)
  DEFINE_GET_ATTR(int, i)
  DEFINE_GET_ATTR(std::string, s)
  DEFINE_GET_ATTR(TensorProto, t)
  DEFINE_GET_ATTR(GraphProto, g)
  DEFINE_GET_ATTRS(float, floats)
  DEFINE_GET_ATTRS(int, ints)
  DEFINE_GET_ATTRS(std::string, strings)
  DEFINE_GET_ATTRS(TensorProto, tensors)
  DEFINE_GET_ATTRS(GraphProto, graphs)

}
