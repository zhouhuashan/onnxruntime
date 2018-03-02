#include "core/framework/op_kernel.h"

namespace Lotus {

    // TODO dummy implementation to facilitate testing
    AllocatorInfo* OpKernelInfo::GetAllocatorInfo() {
        return nullptr;
    }

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
  Status OpKernelInfo::GetAttrs<T>(                                               \
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

    Tensor* OpKernelContext::Output(int index, const TensorShape& shape)
    {
        // In this case, it's assumed that the tensor hasn't been allocated yet,
        // so that it's calling ExecutionFrame to create a tensor in the given position with given shape.
        auto output_arg_index = arg_start_index + static_cast<int>(m_kernel->num_inputs()) + index;
        return m_execution_frame->GetOrCreateTensor(output_arg_index, shape);
    }
}
