#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"

namespace Lotus {

#define DEFINE_GET_ATTR(T, type)                                                  \
  template <>                                                                     \
  Status OpKernelInfo::GetAttr<T>(                                                \
      const std::string& name, T* value) const {                                  \
    const LotusIR::Node& op_def = node();                                         \
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
    const LotusIR::Node& op_def = node();                                         \
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
    DEFINE_GET_ATTR(int64_t, i)
    DEFINE_GET_ATTR(std::string, s)
    DEFINE_GET_ATTR(TensorProto, t)
    DEFINE_GET_ATTR(GraphProto, g)
    DEFINE_GET_ATTRS(float, floats)
    DEFINE_GET_ATTRS(int64_t, ints)
    DEFINE_GET_ATTRS(std::string, strings)
    DEFINE_GET_ATTRS(TensorProto, tensors)
    DEFINE_GET_ATTRS(GraphProto, graphs)

    Tensor* OpKernelContext::output(int index, const TensorShape& shape)
    {
        // In this case, it's assumed that the tensor hasn't been allocated yet,
        // so that it's calling ExecutionFrame to create a tensor in the given position with given shape.
        auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->node().InputDefs().size()) + index;
        return execution_frame_->get_or_create_tensor(output_arg_index, shape);
    }

    // Fetching output tensor without shape is not allowed.
    template<>
    Tensor* OpKernelContext::output<Tensor>(int index) {
        LOTUS_ENFORCE(false, "Please fetch output tensor with specified shape.");
        (index);
        return nullptr;
    }

    OpKernelContext::OpKernelContext(ExecutionFrame* frame, OpKernel* kernel)
        : execution_frame_(frame),
        kernel_(kernel)
    {
        LOTUS_ENFORCE(nullptr != frame && kernel != nullptr);
        arg_start_index_ = frame->get_first_arg_index(kernel->node().Index());
    }
}
