#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "core/framework/op_kernel.h"
#include "onnx/defs/schema.h"
#include "gsl/span"
using namespace onnx;
using namespace Lotus::Common;
namespace Lotus {

#define DEFINE_GET_ATTR(IMPL_T, T, type)                                                       \
  template <>                                                                                  \
  template <>                                                                                  \
  Status OpNodeProtoHelper<IMPL_T>::GetAttr<T>(                                                \
      const std::string& name, T* value) const {                                               \
    const AttributeProto* attr = TryGetAttribute(name);                                        \
    if (!attr) {                                                                               \
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "No attribute with name:'", name, "'is defined."); \
    }                                                                                          \
    if (!attr->has_##type()) {                                                                 \
      return Status(LOTUS, FAIL, "Attibute name and type don't match");                        \
    } else {                                                                                   \
      *value = static_cast<T>(attr->type());                                                   \
      return Status::OK();                                                                     \
    }                                                                                          \
  }

#define DEFINE_GET_ATTRS(IMPL_T, T, list)                                    \
  template <>                                                                \
  template <>                                                                \
  Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                             \
      const std::string& name, std::vector<T>& values) const {               \
    const AttributeProto* attr = TryGetAttribute(name);                      \
    if (!attr) {                                                             \
      return Status(LOTUS, FAIL, "No attribute with this name is defined."); \
    }                                                                        \
    values.reserve(attr->list##_size());                                     \
    for (int i = 0; i < attr->list##_size(); ++i) {                          \
      values.push_back(static_cast<T>(attr->list(i)));                       \
    }                                                                        \
    return Status::OK();                                                     \
  }                                                                          \
  template <>                                                                \
  template <>                                                                \
  Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                             \
      const std::string& name, gsl::span<T> values) const {                  \
    const AttributeProto* attr = TryGetAttribute(name);                      \
    if (!attr) {                                                             \
      return Status(LOTUS, FAIL, "No attribute with this name is defined."); \
    }                                                                        \
    LOTUS_ENFORCE(values.size() == attr->list##_size());                     \
    for (int i = 0; i < attr->list##_size(); ++i) {                          \
      values[i] = static_cast<T>(attr->list(i));                             \
    }                                                                        \
    return Status::OK();                                                     \
  }

#define DEFINE_GET_ATTR_SPECIALIZATIONS(type, list)   \
  DEFINE_GET_ATTR(ProtoHelperNodeContext, type, list) \
  DEFINE_GET_ATTR(InferenceContext, type, list)

#define DEFINE_GET_ATTRS_SPECIALIZATIONS(type, list)   \
  DEFINE_GET_ATTRS(ProtoHelperNodeContext, type, list) \
  DEFINE_GET_ATTRS(InferenceContext, type, list)

DEFINE_GET_ATTR_SPECIALIZATIONS(float, f)
DEFINE_GET_ATTR_SPECIALIZATIONS(int64_t, i)
DEFINE_GET_ATTR_SPECIALIZATIONS(std::string, s)
DEFINE_GET_ATTR_SPECIALIZATIONS(TensorProto, t)
DEFINE_GET_ATTR_SPECIALIZATIONS(GraphProto, g)
DEFINE_GET_ATTRS_SPECIALIZATIONS(float, floats)
DEFINE_GET_ATTRS_SPECIALIZATIONS(int64_t, ints)
DEFINE_GET_ATTRS_SPECIALIZATIONS(std::string, strings)
DEFINE_GET_ATTRS_SPECIALIZATIONS(TensorProto, tensors)
DEFINE_GET_ATTRS_SPECIALIZATIONS(GraphProto, graphs)

size_t ProtoHelperNodeContext::getNumInputs() const {
  return node_.InputDefs().size();
}

size_t ProtoHelperNodeContext::getNumOutputs() const {
  return node_.OutputDefs().size();
}

const AttributeProto* ProtoHelperNodeContext::getAttribute(const std::string& name) const {
  const LotusIR::NodeAttributes& attributes = node_.GetAttributes();
  auto it = attributes.find(name);
  if (it != attributes.end()) {
    return &it->second;
  }
  return nullptr;
}

const TypeProto* ProtoHelperNodeContext::getInputType(size_t index) const {
  return node_.InputDefs()[index]->TypeAsProto();
}

const TypeProto* ProtoHelperNodeContext::getOutputType(size_t index) const {
  return node_.OutputDefs()[index]->TypeAsProto();
}

template <class Impl_t>
uint32_t OpNodeProtoHelper<Impl_t>::GetPrimitiveAttrElementCount(AttributeProto_AttributeType type,
                                                                 const std::string& name) const noexcept {
  const AttributeProto* attr = impl_->getAttribute(name);
  if (attr) {
    switch (type) {
      case AttributeProto_AttributeType_FLOAT:
      case AttributeProto_AttributeType_INT:
      case AttributeProto_AttributeType_STRING:
        return 1;

      case AttributeProto_AttributeType_FLOATS:
        return attr->floats_size();
      case AttributeProto_AttributeType_INTS:
        return attr->ints_size();
      case AttributeProto_AttributeType_STRINGS:
        return attr->strings_size();

        // The following are unsupported through this method
      case AttributeProto_AttributeType_UNDEFINED:
      case AttributeProto_AttributeType_TENSOR:
      case AttributeProto_AttributeType_GRAPH:
      case AttributeProto_AttributeType_TENSORS:
      case AttributeProto_AttributeType_GRAPHS:
      default:
        return 0;
    }
  }

  return 0;
}

template <class Impl_t>
bool OpNodeProtoHelper<Impl_t>::HasPrimitiveAttribute(AttributeProto_AttributeType type,
                                                      const std::string& name) const noexcept {
  return GetPrimitiveAttrElementCount(type, name) > 0;
}

template class OpNodeProtoHelper<ProtoHelperNodeContext>;
template class OpNodeProtoHelper<InferenceContext>;

}  // namespace Lotus
