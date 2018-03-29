#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"

namespace Lotus {

#define DEFINE_GET_ATTR(T, type)                                           \
  template <>                                                              \
  Status OpKernelInfo::GetAttr<T>(                                         \
      const std::string& name, T* value) const {                           \
    const LotusIR::Node& op_def = node();                                  \
    const LotusIR::NodeAttributes attributes = op_def.GetAttributes();     \
    auto it = attributes.find(name);                                       \
    if (it != attributes.end()) {                                          \
      const AttributeProto attr = it->second;                              \
      if (!attr.has_##type()) {                                            \
        return Status(LOTUS, FAIL, "Attibute name and type don't match");  \
      } else {                                                             \
        *value = static_cast<T>(attr.type());                              \
        return Status::OK();                                               \
      }                                                                    \
    }                                                                      \
    return Status(LOTUS, FAIL, "No attribute with this name is defined."); \
  }

#define DEFINE_GET_ATTRS(T, list)                                          \
  template <>                                                              \
  Status OpKernelInfo::GetAttrs<T>(                                        \
      const std::string& name, std::vector<T>& values) const {             \
    const LotusIR::Node& op_def = node();                                  \
    const LotusIR::NodeAttributes attributes = op_def.GetAttributes();     \
    auto it = attributes.find(name);                                       \
    if (it != attributes.end()) {                                          \
      const AttributeProto attr = it->second;                              \
      for (int i = 0; i < attr.list##_size(); ++i) {                       \
        values.push_back(static_cast<T>(attr.list(i)));                    \
      }                                                                    \
      return Status::OK();                                                 \
    }                                                                      \
    return Status(LOTUS, FAIL, "No attribute with this name is defined."); \
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

bool KernelRegistry::VerifyKernelDef(const LotusIR::Node& node, const KernelDef& kernel_def) {
  const LotusIR::OperatorSchema* op_schema = node.Op();
  const size_t len = node.InputArgCount().size();
  if (len > op_schema->GetOpSignature().GetInputs().size()) return false;
  int cur = 0;
  for (size_t input_index = 0; input_index != len; ++input_index) {
    const LotusIR::OpSignature::FormalParameter& param = op_schema->GetOpSignature().GetInputs()[input_index];
    LOTUS_ENFORCE(!param.GetTypeStr().empty());
    const std::unordered_map<std::string, std::vector<MLDataType>>& kernel_type_constraints = kernel_def.TypeConstraints();
    auto allowed_type_list_iter = kernel_type_constraints.find(param.GetTypeStr());
    if (allowed_type_list_iter == kernel_type_constraints.end()) {
      allowed_type_list_iter = kernel_type_constraints.find(param.GetName());
    }
    if (allowed_type_list_iter == kernel_type_constraints.end()) return false;
    for (int i = 0; i < node.InputArgCount()[input_index]; i++)
    {
      LotusIR::NodeArg* arg = node.InputDefs()[cur + i];
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
    cur += node.InputArgCount()[input_index];
  }
  // op_schema may have more inputs than the actual inputs in the node,
  // let's assume all others are optional
  return true;
}

Status KernelRegistry::Register(KernelDefBuilder& kernel_builder,
                                KernelCreateFn kernel_creator) {
  KernelCreateInfo kernel_info(kernel_builder.Build(), kernel_creator);
  auto& op_name = kernel_info.kernel_def->OpName();
  auto& domain = kernel_info.kernel_def->Domain();
  auto& provider_type = kernel_info.kernel_def->Provider();
  int start = 0, end = 0;
  kernel_info.kernel_def->SinceVersion(&start, &end);

  // Check no op version conflicts.
  auto range = kernel_creator_fn_map_.equal_range(op_name);
  for (auto i = range.first; i != range.second; ++i) {
    if (domain == i->second.kernel_def->Domain() &&
        provider_type == i->second.kernel_def->Provider()) {
      int start1 = 0, end1 = 0;
      i->second.kernel_def->SinceVersion(&start1, &end1);
      if (start <= end1 && end >= start1) {
        Status status(LOTUS, FAIL, "Kernels for " + op_name + " have conflicting op versions.");
        return status;
      }
    }
  }

  // Register the kernel.
  kernel_creator_fn_map_.insert({ op_name, kernel_info });
  return Status::OK();
}

Status KernelRegistry::CreateKernel(const LotusIR::OperatorSchema& /*TODO:remove it*/,
                                    const ProviderType& provider_type,
                                    const LotusIR::Node& node,
                                    const AllocatorInfo& allocator_info,
                                    /*out*/ std::unique_ptr<OpKernel>* op_kernel) const {
  // TODO: error check for op_name/op_domain/provider/since_version.
  // TODO: find the real appropriate kernel create info for specific version.
  const LotusIR::OperatorSchema* op_schema = node.Op();
  if (op_schema->GetAttributeParser()) {
    Status status = op_schema->GetAttributeParser()(node.GetAttributes());
    RETURN_IF_ERROR(status);
  }

  auto range = kernel_creator_fn_map_.equal_range(node.OpType());
  for (auto i = range.first; i != range.second; ++i) {
    if (node.Domain() == i->second.kernel_def->Domain() &&
        provider_type == i->second.kernel_def->Provider()) {
      int start, end;
      i->second.kernel_def->SinceVersion(&start, &end);
      int version = 1;  // TODO: Get version from somewhere.
      if (start <= version && version <= end &&
          VerifyKernelDef(node, *i->second.kernel_def)) {
        OpKernelInfo kernel_info(node, allocator_info, *i->second.kernel_def);
        op_kernel->reset(i->second.kernel_create_fn(kernel_info));
        return Status::OK();
      }
    }
  }
  return Status(LOTUS, NOT_IMPLEMENTED, "OP Kernel not found");
}

Tensor* OpKernelContext::output(int index, const TensorShape& shape) {
  // In this case, it's assumed that the tensor hasn't been allocated yet,
  // so that it's calling ExecutionFrame to create a tensor in the given position with given shape.
  auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->node().InputDefs().size()) + index;
  return execution_frame_->get_or_create_tensor(output_arg_index, shape);
}

// Fetching output tensor without shape is not allowed.
template <>
Tensor* OpKernelContext::output<Tensor>(int index) {
  LOTUS_ENFORCE(false, "Please fetch output tensor with specified shape.");
  (index);
  return nullptr;
}

OpKernelContext::OpKernelContext(ExecutionFrame* frame, const OpKernel* kernel)
    : execution_frame_(frame),
      kernel_(kernel) {
  LOTUS_ENFORCE(nullptr != frame && kernel != nullptr);
  arg_start_index_ = frame->get_first_arg_index(kernel->node().Index());
}
}  // namespace Lotus
