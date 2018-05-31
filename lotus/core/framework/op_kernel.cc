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
  const LotusIR::NodeAttributes& attributes = node().GetAttributes();
  auto it = attributes.find(name);
  if (it != attributes.end()) {
    *attribute = &it->second;
    return Status::OK();
  }
  return Status(LOTUS, FAIL, "No attribute with this name is defined.");
}

uint32_t OpKernelInfo::GetPrimitiveAttrElementCount(AttributeProto_AttributeType type,
                                                    const std::string& name) const noexcept {
  const LotusIR::NodeAttributes& attributes = node().GetAttributes();
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

// Find the type that name is bound to in the given node.
// "name" can represent either a type parameter or an input/output parameter.
// Returns null if a match is not found.
const ::onnx::TypeProto* FindTypeBinding(const LotusIR::Node& node, const std::string& name) {
  const OpSchema& op_schema = *node.Op();
  // search inputs:
  const size_t len = node.InputArgCount().size();
  LOTUS_ENFORCE(len <= op_schema.inputs().size());
  int actual_index = 0;
  for (size_t formal_index = 0; formal_index != len; ++formal_index) {
    auto& param = op_schema.inputs()[formal_index];
    if ((param.GetTypeStr() == name) || (param.GetName() == name)) {
      // return type of any corresponding actual parameter, if present
      for (int i = 0, end = node.InputArgCount()[formal_index]; i < end; ++i) {
        const LotusIR::NodeArg* arg = node.InputDefs()[actual_index + i];
        if (!arg->Exists()) continue;  // a missing optional argument
        return arg->TypeAsProto();
      }
    }
    actual_index += node.InputArgCount()[formal_index];
  }
  // search outputs:
  auto& actual_outputs = node.OutputDefs();
  auto num_actual_outputs = actual_outputs.size();
  auto last_formal = op_schema.outputs().size() - 1;
  for (size_t i = 0; i != num_actual_outputs; ++i) {
    const LotusIR::NodeArg* arg = actual_outputs[i];
    if (!arg->Exists()) continue;
    auto& formal = op_schema.outputs()[std::min(i, last_formal)];
    if ((formal.GetTypeStr() == name) || (formal.GetName() == name)) {
      return arg->TypeAsProto();
    }
  }
  return nullptr;
}

// Check whether the types of inputs/outputs of the given node match the extra
// type-constraints of the given kernel. This serves two purposes: first, to
// select the right kernel implementation based on the types of the arguments
// when we have multiple kernels, e.g., Clip<float> and Clip<int>; second, to
// accommodate (and check) mapping of ONNX (specification) type to the Lotus
// implementation type (e.g., if we want to implement ONNX's float16 as a regular
// float in Lotus). (The second, however, requires a globally uniform mapping.)
//
// Note that this is not intended for type-checking the node against the ONNX
// type specification of the corresponding op, which is done before this check.

bool KernelRegistry::VerifyKernelDef(const LotusIR::Node& node,
                                     const KernelDef& kernel_def,
                                     LotusIR::ProviderType exec_provider) {
  // check if domain matches
  if (node.Domain() != kernel_def.Domain())
    return false;

  // check if execution provider matches
  const auto& node_provider = node.GetExecutionProviderType();
  const auto& expected_provider = (node_provider.empty() ? exec_provider : node_provider);
  if (expected_provider != kernel_def.Provider())
    return false;

  // check if version matches
  int kernel_start_version, kernel_end_version;
  kernel_def.SinceVersion(&kernel_start_version, &kernel_end_version);
  int node_version = node.Op()->since_version();
  if (kernel_start_version > node_version || node_version > kernel_end_version)
    return false;

  // check if type matches
  auto& kernel_type_constraints = kernel_def.TypeConstraints();
  for (auto& constraint : kernel_type_constraints) {
    const std::string& name = constraint.first;
    const std::vector<MLDataType>& allowed_types = constraint.second;
    const ::onnx::TypeProto* actual_type = FindTypeBinding(node, name);

    // If actual_type is null, this represents a type-constraint on a
    // missing optional parameter, which can be skipped.
    // TODO: We should check that names specified in kernel_type_constraints are
    // valid names (of types or parameters) at the time that kernels are registered.

    if ((nullptr != actual_type) &&
        !std::any_of(allowed_types.begin(), allowed_types.end(),
                     [actual_type](const MLDataType& expected_type) {
                       return expected_type->IsCompatible(*actual_type);
                     })) {
      return false;
    }
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
                                    const IExecutionProvider* execution_provider,
                                    /*out*/ std::unique_ptr<OpKernel>* op_kernel) const {
  auto range = kernel_creator_fn_map_.equal_range(node.OpType());
  for (auto i = range.first; i != range.second; ++i) {
    // Check if the kernel is ill-formed.
    if (!i->second.status.IsOK()) {
      return i->second.status;
    }
    if (VerifyKernelDef(node, *i->second.kernel_def)) {
      OpKernelInfo kernel_info(node, *i->second.kernel_def, execution_provider);
      op_kernel->reset(i->second.kernel_create_func(kernel_info));
      return Status::OK();
    }
  }

  if (create_func_kernel_) {
    // The node is assigned to an execution provider and no kernel registered
    // for the operator referred by the node. Create FunctionKernel to delegate
    // the node run to corresponding execution provider.
    auto& kernelCreatorInfo = kernel_creator_fn_map_.find(LotusIR::kFunctionOp)->second;
    OpKernelInfo kernel_info(node, *kernelCreatorInfo.kernel_def, execution_provider);
    op_kernel->reset(kernelCreatorInfo.kernel_create_func(kernel_info));
    return Status::OK();
  } else
    return Status(LOTUS, FAIL, "Kernel not found.");
}

bool KernelRegistry::CanExecutionProviderCreateKernel(const LotusIR::Node& node, LotusIR::ProviderType exec_provider) const {
  auto range = kernel_creator_fn_map_.equal_range(node.OpType());
  LOTUS_ENFORCE(node.GetExecutionProviderType().empty());
  for (auto i = range.first; i != range.second; ++i) {
    if (i->second.status.IsOK() &&
        VerifyKernelDef(node, *i->second.kernel_def, exec_provider)) {
      return true;
    }
  }
  return false;
}

Tensor* OpKernelContext::Output(int index, const TensorShape& shape) {
  if (index >= kernel_->Node().OutputDefs().size())
    return nullptr;
  // In this case, it's assumed that the tensor hasn't been allocated yet,
  // so that it's calling ExecutionFrame to create a tensor in the given position with given shape.
  auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->Node().InputDefs().size()) + index;
  MLValueAllocationParameters parameters;
  parameters.tensor_shape = shape;
  Tensor* ret;
  Status status = execution_frame_->GetOrCreateMLValue<Tensor>(output_arg_index, parameters, ret);
  LOTUS_ENFORCE(status.IsOK());
  return ret;
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
