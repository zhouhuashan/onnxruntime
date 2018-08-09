#pragma once

#include <functional>

#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/execution_frame.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/tensor.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "onnx/defs/schema.h"
#include "gsl/span"

#include <functional>

using namespace LotusIR;

class IMLOpKernel;

namespace Lotus {
class OpKernelContext;
class OpKernelWrapper;

class OpKernel {
 public:
  using DoneCallback = std::function<void()>;

  explicit OpKernel(const OpKernelInfo& info) : op_kernel_info_(info) {}
  virtual ~OpKernel() = default;

  const LotusIR::Node& Node() const {
    return op_kernel_info_.node();
  }

  const Lotus::KernelDef& KernelDef() const {
    return op_kernel_info_.GetKernelDef();
  }

  virtual Status Compute(OpKernelContext* context) const = 0;

  virtual Status ComputeAsync(OpKernelContext*,
                              DoneCallback) const {
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  const AllocatorInfo& Allocator(MemType mem_type) const {
    return op_kernel_info_.GetAllocatorInfo(mem_type);
  }

  const OpKernelInfo& Info() const { return op_kernel_info_; }

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(OpKernel);
  OpKernelInfo op_kernel_info_;
};

class OpKernelContext {
 public:
  typedef std::unordered_map<std::string, size_t> ArgMap;

  explicit OpKernelContext(ExecutionFrame* frame,
                           const OpKernel* kernel,
                           const Logging::Logger& logger);

  ~OpKernelContext() = default;

  /** 
  Return the number of inputs for a variadic argument. 
  @param arg_num The operator argument number. 
  @returns Number of inputs the argument has. 
  */
  int NumInputs(size_t arg_num) const {
    auto& arg_counts = kernel_->Node().InputArgCount();

    LOTUS_ENFORCE(arg_num < arg_counts.size(),
                  "Invalid arg_num of ", arg_num, ". Num args is ", arg_counts.size());

    return arg_counts[arg_num];
  }

  template <typename T>
  const T* Input(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= kernel_->Node().InputDefs().size())
      return nullptr;

    return execution_frame_->GetValue<T>(arg_start_index_ + index);
  }

  MLDataType InputType(int index) const {
    return execution_frame_->GetType(arg_start_index_ + index);
  }

  MLDataType OutputType(int index) const {
    auto output_arg_index = GetOutputArgIndex(index);
    return execution_frame_->GetType(output_arg_index);
  }

  // Fetch output (non-tensor) with specified index.
  template <typename T>
  T* Output(int index) {
    if (index < 0 || static_cast<size_t>(index) >= kernel_->Node().OutputDefs().size())
      return nullptr;

    auto output_arg_index = GetOutputArgIndex(index);
    MLValueAllocationParameters parameters;
    T* ret = nullptr;
    LOTUS_ENFORCE(execution_frame_->GetOrCreateMLValue<T>(output_arg_index, parameters, ret).IsOK());
    return ret;
  }

  int GetOutputArgIndex(int index) const {
    return arg_start_index_ + static_cast<int>(kernel_->Node().InputDefs().size()) + index;
  }

  // In the case that memory allocation has not been done for an output tensor,
  // The memory allocation will be done on-the-fly with given tensor shape.
  // Return nullptr if the output is an unused optional output.
  Tensor* Output(int index, const TensorShape& shape);

  const Logging::Logger& Logger() const {
    return *logger_;
  }

  int InputCount() const {
    return static_cast<int>(kernel_->Node().InputDefs().size());
  }

  int OutputCount() const {
    return static_cast<int>(kernel_->Node().OutputDefs().size());
  }

  Status GetTempSpaceAllocator(AllocatorPtr* output) const {
    *output = execution_frame_->GetAllocator(kernel_->Allocator(kMemTypeDefault));
    if (!*output)
      return Status(Common::LOTUS, Common::FAIL, "TempSpace allocator not found");
    return Status::OK();
  }

  Fence_t InputFence(int index) const {
    if (index >= InputCount())
      return nullptr;

    return execution_frame_->GetFence(arg_start_index_ + index);
  }

  Fence_t OutputFence(int index) const {
    if (index >= OutputCount())
      return nullptr;

    return execution_frame_->GetFence(arg_start_index_ + InputCount() + index);
  }

 private:
  ExecutionFrame* execution_frame_ = nullptr;
  const OpKernel* kernel_ = nullptr;
  const Logging::Logger* logger_ = nullptr;

  // The argument starting index in ExecutionFrame.
  int arg_start_index_ = -1;
};

// Fetching output tensor without shape is not allowed.
template <>
inline Tensor* OpKernelContext::Output<Tensor>(int) {
  LOTUS_ENFORCE(false, "Please fetch output tensor with specified shape.");
  return nullptr;
}

using KernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;

struct KernelCreateInfo {
  std::unique_ptr<KernelDef> kernel_def;  // Owned and stored in the global kernel registry.
  KernelCreateFn kernel_create_func;
  Status status;

  KernelCreateInfo(std::unique_ptr<KernelDef> definition,
                   KernelCreateFn create_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func) {}

  KernelCreateInfo(KernelCreateInfo&& other)
      : kernel_def(std::move(other.kernel_def)),
        kernel_create_func(other.kernel_create_func) {}
};

using KernelCreateMap = std::multimap<std::string, KernelCreateInfo>;

class KernelRegistry {
 public:
  // Register a kernel with kernel definition and function to create the kernel.
  Status Register(KernelDefBuilder& kernel_def_builder, KernelCreateFn kernel_creator);

  // Mainly for provide debug info
  std::vector<std::string> GetAllRegisteredOpNames() const;

  // factory functions should always return a unique_ptr for maximum flexibility
  // for its clients unless the factory is managing the lifecycle of the pointer
  // itself.
  // TODO(Task:132) Make usage of unique_ptr/shared_ptr as out param consistent
  Status CreateKernel(const LotusIR::Node& node,
                      const IExecutionProvider* execution_provider,
                      const SessionState& session_state,
                      std::unique_ptr<OpKernel>* op_kernel) const;

  Status SearchKernelRegistry(const LotusIR::Node& node,
                              /*out*/ const KernelCreateInfo** kernel_create_info) const;

  // check if a execution provider can create kernel for a node
  bool CanExecutionProviderCreateKernel(
      const LotusIR::Node& node,
      LotusIR::ProviderType exec_provider) const;

  KernelRegistry() : KernelRegistry([](std::function<void(KernelCreateInfo&&)>) {}) {}

  KernelRegistry(std::function<void(std::function<void(KernelCreateInfo&&)>)> kernel_reg_fn)
      : kernel_reg_fn_(kernel_reg_fn) {}

  void RegisterInternal(KernelCreateInfo& create_info) const;

 private:
  friend class InferenceSession;

  // Check if the node's input/outpuData/attributes are compatible with this
  // kernel_def, If so, the kernel defined by the kernel_def is used to
  // execute this node. exec_provider is used to match kernel when node has no provider
  static bool VerifyKernelDef(const LotusIR::Node& node,
                              const KernelDef& kernel_def,
                              std::string& error_str,
                              LotusIR::ProviderType exec_provider = "");

  // Kernel create function map from op name to kernel creation info.
  mutable std::unique_ptr<KernelCreateMap> kernel_creator_fn_map_ =
      std::make_unique<KernelCreateMap>();
  KernelCreateMap const& kernel_creator_fn_map() const;
  mutable std::once_flag kernelCreationFlag;

  std::function<void(std::function<void(KernelCreateInfo&&)>)> kernel_reg_fn_;
};

// Forward declarations for the non-specialized BuildKernel method.
template <typename T>
KernelCreateInfo BuildKernel();

namespace ML {
template <typename T>
KernelCreateInfo BuildKernel();
}  // namespace ML

namespace Cuda {
template <typename T>
KernelCreateInfo BuildKernel();
}  // namespace Cuda

namespace MklDnn {
template <typename T>
KernelCreateInfo BuildKernel();
}  // namespace MklDnn

// Naming convention for operator kernel classes
#define ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name) \
  provider##_##name##_##domain##_ver##ver

#define ONNX_CPU_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_ML_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kMLDomain, ver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)            \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                 \
  template <>                                                                         \
  KernelCreateInfo                                                                    \
  BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {       \
    return KernelCreateInfo(                                                          \
        builder.SetName(#name)                                                        \
            .SetDomain(domain)                                                        \
            .SinceVersion(ver)                                                        \
            .Provider(provider)                                                       \
            .Build(),                                                                 \
        [](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); }); \
  }

#define ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name) \
  provider##_##name##_##domain##_ver##startver##_##endver

#define ONNX_CPU_OPERATOR_VERSIONED_KERNEL(name, startver, endver, builder, ...) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, kOnnxDomain, startver, endver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_ML_KERNEL(name, startver, endver, builder, ...) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, kMLDomain, startver, endver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, provider, builder, ...)      \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name);           \
  template <>                                                                                          \
  KernelCreateInfo                                                                                     \
  BuildKernel<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name)>() { \
    return KernelCreateInfo(                                                                           \
        builder.SetName(#name)                                                                         \
            .SetDomain(domain)                                                                         \
            .SinceVersion(startver, endver)                                                            \
            .Provider(provider)                                                                        \
            .Build(),                                                                                  \
        [](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); });                  \
  }

#define ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name) \
  provider##_##name##_##domain##_ver##ver##_##type

#define ONNX_CPU_OPERATOR_TYPED_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kOnnxDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMLDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...)      \
  class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name);           \
  template <>                                                                               \
  KernelCreateInfo                                                                          \
  BuildKernel<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>() { \
    return KernelCreateInfo(                                                                \
        builder.SetName(#name)                                                              \
            .SetDomain(domain)                                                              \
            .SinceVersion(ver)                                                              \
            .Provider(provider)                                                             \
            .Build(),                                                                       \
        [](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); });       \
  }

}  // namespace Lotus
