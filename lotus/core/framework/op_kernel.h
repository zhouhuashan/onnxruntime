#pragma once

#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/execution_frame.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "onnx/defs/schema.h"
#include "gsl/span"

class IMLOpKernel;

namespace Lotus {
class OpKernelContext;
class OpKernelWrapper;

// A very light-weight class, which works as an aggregated
// view of all data needed for constructing a Kernel instance.
// NOTE: it does not own/hold any objects.
class OpKernelInfo {
 public:
  explicit OpKernelInfo(const LotusIR::Node& node,
                        const AllocatorInfo& allocator_info,
                        const KernelDef& kernel_def,
                        const IExecutionProvider* execution_provider)
      : node_(node),
        allocator_info_(allocator_info),
        kernel_def_(kernel_def),
        execution_provider_(execution_provider) {}

  //Get a single attribute
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  //Get repeated attributes
  template <typename T>
  Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  template <typename T>
  Status GetAttrs(const std::string& name, gsl::span<T> values) const;

  uint32_t GetPrimitiveAttrElementCount(AttributeProto_AttributeType type,
                                        const std::string& name) const noexcept;

  bool HasPrimitiveAttribute(AttributeProto_AttributeType type,
                             const std::string& name) const noexcept;
  
  const LotusIR::Node& node() const {
    return node_;
  }

  const AllocatorInfo& GetAllocatorInfo() const {
    return allocator_info_;
  }

  const KernelDef& GetKernelDef() const {
    return kernel_def_;
  }

  const IExecutionProvider* GetExecutionProvider() const {
    return execution_provider_;
  }

  Status GetAttributeProto(const std::string& name,
                           const AttributeProto** attribute) const;

 private:
  const LotusIR::Node& node_;
  const AllocatorInfo& allocator_info_;
  const KernelDef& kernel_def_;
  // For non cpu/cuda case, this pointer should be set so that function kernel
  // will delegate kernel compute call to <execution_provider> compute call.
  const Lotus::IExecutionProvider* execution_provider_;
};

class OpKernel {
 public:
  typedef std::function<void()> DoneCallback;

  explicit OpKernel(const OpKernelInfo& info) : op_kernel_info_(info) {
  }

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

  const AllocatorInfo& Allocator() const {
    return op_kernel_info_.GetAllocatorInfo();
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

  ~OpKernelContext(){};

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
    return execution_frame_->GetValue<T>(arg_start_index_ + index);
  }

  MLDataType InputType(int index) const {
    return execution_frame_->GetType(arg_start_index_ + index);
  }

  // Fetch output (non-tensor) with specified index.
  template <typename T>
  T* Output(int index) {
    auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->Node().InputDefs().size()) + index;
    MLValueAllocationParameters paramerters;
    return execution_frame_->GetOrCreateMLValue<T>(output_arg_index, paramerters);
  }

  // In the case that memory allocation has not been done for an output tensor,
  // The memory allocation will be done on-the-fly with given tensor shape.
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

 private:
  ExecutionFrame* execution_frame_ = nullptr;
  const OpKernel* kernel_ = nullptr;
  const Logging::Logger* logger_ = nullptr;

  // The argument starting index in ExecutionFrame.
  int arg_start_index_ = -1;
};

typedef OpKernel* (*KernelCreateFn)(const OpKernelInfo& info);
typedef IMLOpKernel* (*IMLOpKernelCreateFn)();

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
                      const AllocatorInfo& allocator_info,
                      const IExecutionProvider* execution_provider,
                      std::unique_ptr<OpKernel>* op_kernel) const;

  static KernelRegistry& Instance() {
    static KernelRegistry kernel_registry;
    return kernel_registry;
  }

 private:
  KernelRegistry() = default;

  struct KernelCreateInfo {
    unique_ptr<KernelDef> kernel_def;  // Owned and stored in the global kernel registry.
    KernelCreateFn kernel_create_func;
    Status status;

    KernelCreateInfo(unique_ptr<KernelDef> definition,
                     KernelCreateFn create_func)
        : kernel_def(std::move(definition)),
          kernel_create_func(create_func) { }

    KernelCreateInfo(KernelCreateInfo&& other)
        : kernel_def(std::move(other.kernel_def)),
          kernel_create_func(other.kernel_create_func) { }
  };

  // Check if the node's input/outpuData/attributes are compatible with this
  // kernel_def, If so, the kernel defined by the kernel_def is used to
  // execute this node.
  static bool VerifyKernelDef(const LotusIR::Node& node,
                              const KernelDef& kernel_def);

  // Kernel create function map from op name to kernel creation info.
  std::multimap<std::string, KernelCreateInfo> kernel_creator_fn_map_;
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


#define REGISTER_ABI_KERNEL(kernel_def_builder, ...) \
  REGISTER_ABI_KERNEL_UNIQ_HELPER(__COUNTER__, kernel_def_builder, __VA_ARGS__)

#define REGISTER_ABI_KERNEL_UNIQ_HELPER(counter, kernel_def_builder, ...) \
  REGISTER_ABI_KERNEL_UNIQ(counter, kernel_def_builder, __VA_ARGS__)

#define REGISTER_ABI_KERNEL_UNIQ(counter, kernel_def_builder, ...)                                \
  static Lotus::Common::Status kernel_def_builder_##counter##_status =                            \
      KernelRegistry::Instance().Register(                                                        \
          kernel_def_builder,                                                                     \
          [](const OpKernelInfo& info)                                                            \
              -> OpKernel* {                                                                      \
            auto create_op_kernel = []() -> IMLOpKernel* { return new MLOpKernel<__VA_ARGS__>(); }; \
            return new AbiOpKernel(create_op_kernel, info);                                         \
          });

}  // namespace Lotus
