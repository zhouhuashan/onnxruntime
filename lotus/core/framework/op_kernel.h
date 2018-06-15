#pragma once

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

class IMLOpKernel;

namespace Lotus {
class OpKernelContext;
class OpKernelWrapper;

class OpKernel {
 public:
  using DoneCallback = std::function<void()>;

  explicit OpKernel(const OpKernelInfo& info) : op_kernel_info_(info) {
  }
  virtual ~OpKernel() = default;

  const LotusIR::Node& Node() const {
    return op_kernel_info_.node();
  }

  const Lotus::KernelDef& KernelDef() const {
    return op_kernel_info_.GetKernelDef();
  }

  virtual Status Compute(OpKernelContext* context) const = 0;

  virtual Status ComputeAsync(OpKernelContext* context,
                              DoneCallback done) const {
    UNUSED_PARAMETER(context);
    UNUSED_PARAMETER(done);
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  const AllocatorInfo& Allocator(MemType mem_type) const {
    return op_kernel_info_.GetAllocatorInfo(mem_type);
  }

 protected:
  OpKernelInfo op_kernel_info_;
};

class OpKernelContext {
 public:
  typedef std::unordered_map<std::string, size_t> ArgMap;

  explicit OpKernelContext(ExecutionFrame* frame,
                           const OpKernel* kernel,
                           const Logging::Logger& logger);

  ~OpKernelContext() = default;
  ;

  /** 
  Return the number of inputs for a variadic argument. 
  @param arg_num The operator argument number. 
  @returns Number of inputs the argument has. 
  */
  int NumInputs(int arg_num) const {
    auto& arg_counts = kernel_->Node().InputArgCount();

    LOTUS_ENFORCE(arg_num < arg_counts.size(),
                  "Invalid arg_num of ", arg_num, ". Num args is ", arg_counts.size());

    return arg_counts[arg_num];
  }

  template <typename T>
  const T* Input(int index) const {
    if (index >= kernel_->Node().InputDefs().size())
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
    if (index >= kernel_->Node().OutputDefs().size())
      return nullptr;

    auto output_arg_index = GetOutputArgIndex(index);
    MLValueAllocationParameters parameters;
    T* ret;
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
      return Status(LOTUS, FAIL, "TempSpace allocator not found");
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
inline Tensor* OpKernelContext::Output<Tensor>(int index) {
  LOTUS_ENFORCE(false, "Please fetch output tensor with specified shape.");
  (index);
  return nullptr;
}

using KernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;

class KernelRegistry {
 public:
  struct KernelCreateInfo {
    unique_ptr<KernelDef> kernel_def;  // Owned and stored in the global kernel registry.
    KernelCreateFn kernel_create_func;
    Status status;

    KernelCreateInfo(unique_ptr<KernelDef> definition,
                     KernelCreateFn create_func)
        : kernel_def(std::move(definition)),
          kernel_create_func(create_func) {}

    KernelCreateInfo(KernelCreateInfo&& other)
        : kernel_def(std::move(other.kernel_def)),
          kernel_create_func(other.kernel_create_func) {}
  };

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

  static KernelRegistry& Instance() {
    static KernelRegistry kernel_registry(true);
    return kernel_registry;
  }

 protected:
  KernelRegistry(bool create_func_kernel_flag) : create_func_kernel_(create_func_kernel_flag) {}

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
  std::multimap<std::string, KernelCreateInfo> kernel_creator_fn_map_;

  bool create_func_kernel_;
};

#define REGISTER_KERNEL(kernel_def_builder, ...) \
  REGISTER_KERNEL_UNIQ_HELPER(__COUNTER__, kernel_def_builder, __VA_ARGS__)

#define REGISTER_KERNEL_UNIQ_HELPER(counter, kernel_def_builder, ...) \
  REGISTER_KERNEL_UNIQ(counter, kernel_def_builder, __VA_ARGS__)

#define REGISTER_KERNEL_UNIQ(counter, kernel_def_builder, ...)         \
  static Lotus::Common::Status kernel_def_builder_##counter##_status = \
      KernelRegistry::Instance().Register(                             \
          kernel_def_builder,                                          \
          [](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })

}  // namespace Lotus
