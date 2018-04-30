#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/graph/op.h"
#include "op_kernel_abi_wrapper.h"
namespace Lotus {

#define DEFINE_GET_ATTR(T, type)                                        \
  template <>                                                           \
  Status OpKernelInfo::GetAttr<T>(                                      \
      const std::string& name, T* value) const {                        \
    const AttributeProto* attr = nullptr;                               \
    RETURN_IF_ERROR(GetAttributeProto(name, &attr));                    \
    if (!attr->has_##type()) {                                          \
      return Status(LOTUS, FAIL, "Attibute name and type don't match"); \
    } else {                                                            \
      *value = static_cast<T>(attr->type());                            \
      return Status::OK();                                              \
    }                                                                   \
  }

#define DEFINE_GET_ATTRS(T, list)                              \
  template <>                                                  \
  Status OpKernelInfo::GetAttrs<T>(                            \
      const std::string& name, std::vector<T>& values) const { \
    const AttributeProto* attr = nullptr;                      \
    RETURN_IF_ERROR(GetAttributeProto(name, &attr));           \
    values.reserve(attr->list##_size());                       \
    for (int i = 0; i < attr->list##_size(); ++i) {            \
      values.push_back(static_cast<T>(attr->list(i)));         \
    }                                                          \
    return Status::OK();                                       \
  }                                                            \
  template <>                                                  \
  Status OpKernelInfo::GetAttrs<T>(                            \
      const std::string& name, gsl::span<T> values) const {    \
    const AttributeProto* attr = nullptr;                      \
    Status status = GetAttributeProto(name, &attr);            \
    if (!status.IsOK()) {                                      \
      return status;                                           \
    }                                                          \
    LOTUS_ENFORCE(values.size() == attr->list##_size());       \
    for (int i = 0; i < attr->list##_size(); ++i) {            \
      values[i] = static_cast<T>(attr->list(i));               \
    }                                                          \
    return Status::OK();                                       \
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

Status OpKernelInfo::GetAttributeProto(const std::string& name,
                                       const AttributeProto** attribute) const {
  const LotusIR::NodeAttributes &attributes = node().GetAttributes();
  auto it = attributes.find(name);
  if (it != attributes.end()) {
    *attribute = &it->second;
    return Status::OK();
  }
  return Status(LOTUS, FAIL, "No attribute with this name is defined.");
}

uint32_t OpKernelInfo::GetPrimitiveAttrElementCount(AttributeProto_AttributeType type,
                                                    const std::string& name) const noexcept {
  const LotusIR::NodeAttributes &attributes = node().GetAttributes();
  auto it = attributes.find(name);
  if (it != attributes.end()) {
    const AttributeProto attr = it->second;
    switch (type) {
      case AttributeProto_AttributeType_FLOAT:
      case AttributeProto_AttributeType_INT:
      case AttributeProto_AttributeType_STRING:
        return 1;

      case AttributeProto_AttributeType_FLOATS:
        return attr.floats_size();
      case AttributeProto_AttributeType_INTS:
        return attr.ints_size();
      case AttributeProto_AttributeType_STRINGS:
        return attr.strings_size();

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

bool OpKernelInfo::HasPrimitiveAttribute(AttributeProto_AttributeType type,
                                         const std::string& name) const noexcept {
  return GetPrimitiveAttrElementCount(type, name) > 0;
}

std::vector<std::string> KernelRegistry::GetAllRegisteredOpNames() const {
  std::vector<std::string> ret(kernel_creator_fn_map_.size());
  size_t i = 0;
  for (const auto& kvp : kernel_creator_fn_map_) {
    ret[i++] = kvp.first;
  }
  return ret;
}

bool KernelRegistry::VerifyKernelDef(const LotusIR::Node& node,
                                     const KernelDef& kernel_def) {
  const OpSchema& op_schema = *node.Op();
  const size_t len = node.InputArgCount().size();
  if (len > op_schema.inputs().size()) return false;
  int cur = 0;
  for (size_t input_index = 0; input_index != len; ++input_index) {
    auto& param = op_schema.inputs()[input_index];
    LOTUS_ENFORCE(!param.GetTypeStr().empty());
    //param.type_str_ could be a real type string(e.g. int32), or a symbolic one(e.g. T)
    //If it's a real type string, we check exact match at there
    //Otherwise, there should be an entry in the type_constraints_ of this kernel_def
    //  1. Either with this param name
    //  2. Or with this type_str_
    auto& kernel_type_constraints = kernel_def.TypeConstraints();
    auto allowed_type_list_iter = kernel_type_constraints.find(param.GetTypeStr());
    if (allowed_type_list_iter == kernel_type_constraints.end()) {
      allowed_type_list_iter = kernel_type_constraints.find(param.GetName());
    }
    if (allowed_type_list_iter == kernel_type_constraints.end()) {
      for (int i = 0; i < node.InputArgCount()[input_index]; i++) {
        const LotusIR::NodeArg* arg = node.InputDefs()[cur + i];
        if (!arg->Exists()) continue;  //It's an optional arg in the middle of the input list
        onnx::DataType real_type = arg->Type();
        if (*real_type != param.GetTypeStr()) {
          return false;
        }
      }
    } else {
      for (int i = 0; i < node.InputArgCount()[input_index]; i++) {
        const LotusIR::NodeArg* arg = node.InputDefs()[cur + i];
        if (!arg->Exists()) continue;  //It's an optional arg in the middle of the input list
        const ::onnx::TypeProto& real_type = arg->ToProto().type();
        if (!std::any_of(allowed_type_list_iter->second.begin(),
                         allowed_type_list_iter->second.end(),
                         [real_type](const MLDataType& expected_type) {
                           return expected_type->IsCompatible(real_type);
                         })) {
          return false;
        }
      }
    }
    cur += node.InputArgCount()[input_index];
  }
  return true;
}

Status KernelRegistry::Register(KernelDefBuilder& kernel_builder,
                                KernelCreateFn kernel_creator) {
  KernelCreateInfo create_info(kernel_builder.Build(), kernel_creator);
  auto& op_name = create_info.kernel_def->OpName();
  auto& domain = create_info.kernel_def->Domain();
  auto& provider_type = create_info.kernel_def->Provider();
  int start = 0, end = 0;
  create_info.kernel_def->SinceVersion(&start, &end);

  // Check op version conflicts.
  auto range = kernel_creator_fn_map_.equal_range(op_name);
  for (auto i = range.first; i != range.second; ++i) {
    if (domain == i->second.kernel_def->Domain() &&
        provider_type == i->second.kernel_def->Provider()) {
      int start1 = 0, end1 = 0;
      i->second.kernel_def->SinceVersion(&start1, &end1);
      if (start <= end1 && end >= start1) {
        create_info.status =
          Status(LOTUS, FAIL,
                 "Failed to add kernel for " + op_name +
                 ": Conflicting with a registered kernel with op versions [" +
                 std::to_string(start1) + "," + std::to_string(end1) + "].");
        return create_info.status;
      }
    }
  }

  // Register the kernel.
  // Ownership of the KernelDef is transferred to the map.
  kernel_creator_fn_map_.emplace(op_name, std::move(create_info));
  return Status::OK();
}

Status KernelRegistry::CreateKernel(const LotusIR::Node& node,
                                    const AllocatorInfo& allocator_info,
                                    const IExecutionProvider* execution_provider,
                                    /*out*/ std::unique_ptr<OpKernel>* op_kernel) const {
  auto range = kernel_creator_fn_map_.equal_range(node.OpType());
  for (auto i = range.first; i != range.second; ++i) {
    // Check if the kernel is ill-formed.
    if (!i->second.status.IsOK()) {
      return i->second.status;
    }
    if (node.Domain() == i->second.kernel_def->Domain() &&
        node.GetExecutionProvider() == i->second.kernel_def->Provider()) {
      int start, end;
      i->second.kernel_def->SinceVersion(&start, &end);
      int version = node.Op()->since_version();
      if (start <= version && version <= end &&
          VerifyKernelDef(node, *i->second.kernel_def)) {
        OpKernelInfo kernel_info(node, allocator_info, *i->second.kernel_def, execution_provider);
        op_kernel->reset(i->second.kernel_create_func(kernel_info));
        return Status::OK();
      }
    }
  }

  // The node is assigned to an execution provider and no kernel registered
  // for the operator referred by the node. Create FunctionKernel to delegate
  // the node run to corresponding execution provider.
  auto& kernelCreatorInfo = kernel_creator_fn_map_.find(LotusIR::kFunctionOp)->second;
  OpKernelInfo kernel_info(node, allocator_info, *kernelCreatorInfo.kernel_def, execution_provider);
  op_kernel->reset(kernelCreatorInfo.kernel_create_func(kernel_info));
  return Status::OK();
}

Tensor* OpKernelContext::Output(int index, const TensorShape& shape) {
  // In this case, it's assumed that the tensor hasn't been allocated yet,
  // so that it's calling ExecutionFrame to create a tensor in the given position with given shape.
  auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->Node().InputDefs().size()) + index;
  MLValueAllocationParameters parameters;
  parameters.tensor_shape = shape;
  return execution_frame_->GetOrCreateMLValue<Tensor>(output_arg_index, parameters);
}

// Fetching output tensor without shape is not allowed.
template <>
Tensor* OpKernelContext::Output<Tensor>(int index) {
  LOTUS_ENFORCE(false, "Please fetch output tensor with specified shape.");
  (index);
  return nullptr;
}

OpKernelContext::OpKernelContext(ExecutionFrame* frame,
                                 const OpKernel* kernel,
                                 const Logging::Logger& logger)
    : execution_frame_(frame),
      kernel_(kernel),
      logger_(&logger) {
  LOTUS_ENFORCE(frame != nullptr, "Execution frame was null");
  LOTUS_ENFORCE(kernel != nullptr, "OpKernel was null");

  arg_start_index_ = frame->GetFirstArgIndex(kernel->Node().Index());
}

}  // namespace Lotus
