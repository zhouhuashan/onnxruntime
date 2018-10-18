// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/common/exceptions.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "gsl/span"
#include "onnx/defs/schema.h"

namespace onnxruntime {
class ExecutionFrame;
class OpKernelContext;
class OpKernelWrapper;

class OpKernel {
 public:
  using DoneCallback = std::function<void()>;

  explicit OpKernel(const OpKernelInfo& info) : op_kernel_info_(info) {}
  virtual ~OpKernel() = default;

  const onnxruntime::Node& Node() const {
    return op_kernel_info_.node();
  }

  const ::onnxruntime::KernelDef& KernelDef() const {
    return op_kernel_info_.GetKernelDef();
  }

  virtual Status Compute(OpKernelContext* context) const = 0;

  virtual Status ComputeAsync(OpKernelContext*,
                              DoneCallback) const {
    ONNXRUNTIME_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  const ONNXRuntimeAllocatorInfo& Allocator(ONNXRuntimeMemType mem_type) const {
    return op_kernel_info_.GetAllocatorInfo(mem_type);
  }

  const OpKernelInfo& Info() const { return op_kernel_info_; }

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpKernel);
  OpKernelInfo op_kernel_info_;
};

class OpKernelContext {
  // See explicit Tensor specialization below
  template <typename Result, typename TReg>
  struct Fetcher {
    static const Result* Input(const OpKernelContext& ctx, int index) {
      const MLValue* p_ml_value = ctx.GetInputMLValue(index);
      return p_ml_value ? &(p_ml_value->Get<TReg>()) : nullptr;
    }
    static Result* Output(OpKernelContext& ctx, int index) {
      if (index < 0 || index >= ctx.OutputCount())
        return nullptr;
      MLValue* p_ml_value = nullptr;
      ONNXRUNTIME_ENFORCE(ctx.GetOrCreateOutputMLValue(index, p_ml_value).IsOK());
      return p_ml_value ? p_ml_value->GetMutable<TReg>() : nullptr;
    }
  };

  template <typename T, typename... Types>
  struct TypeRegistrationDispatcher;

  template <typename T>
  struct TypeRegistrationDispatcher<T> : public Fetcher<T, T> {
  };

  template <typename T, typename... Params>
  struct TypeRegistrationDispatcher<TypeRegister<T, Params...>> : public Fetcher<T, TypeRegister<T, Params...>> {
  };

  template <typename T, const char D[], const char N[], typename... Params>
  struct TypeRegistrationDispatcher<OpaqueRegister<T, D, N, Params...>> : public Fetcher<T, OpaqueRegister<T, D, N, Params...>> {
  };

 public:
  using ArgMap = std::unordered_map<std::string, size_t>;

  explicit OpKernelContext(ExecutionFrame* frame,
                           const OpKernel* kernel,
                           const logging::Logger& logger);

  virtual ~OpKernelContext() = default;

  /**
  Return the number of inputs for a variadic argument.
  @param arg_num The operator argument number.
  @returns Number of inputs the argument has.
  */
  int NumVariadicInputs(size_t arg_num) const;

  MLDataType InputType(int index) const;
  MLDataType OutputType(int index) const;

  template <typename T>
  const auto* Input(int index) const {
    return TypeRegistrationDispatcher<T>::Input(*this, index);
  }

  // Fetch output (non-tensor) with specified index.
  template <typename T>
  auto* Output(int index) {
    return TypeRegistrationDispatcher<T>::Output(*this, index);
  }

  // In the case that memory allocation has not been done for an output tensor,
  // The memory allocation will be done on-the-fly with given tensor shape.
  // Return nullptr if the output is an unused optional output.
  Tensor* Output(int index, const TensorShape& shape);

  const logging::Logger& Logger() const {
    return *logger_;
  }

  int InputCount() const {
    return static_cast<int>(kernel_->Node().InputDefs().size());
  }

  int OutputCount() const {
    return static_cast<int>(kernel_->Node().OutputDefs().size());
  }

  Status GetTempSpaceAllocator(AllocatorPtr* output) const;

  /**
  Return the fence of current node's input.
  @param index The index of the input.
  @returns Point to the Fence of the input MLValue.
  It is null if the input MLValue doesn't have fence or the input is optional.
  */
  Fence_t InputFence(int index) const;

  /**
  Return the fence of current node's output identifed by index.
  @param index The index of the output.
  @returns Point to the Fence of the output MLValue.
  It is null if the output MLValue doesn't have fence or the output is optional.
  */
  Fence_t OutputFence(int index) const;

 protected:
  onnxruntime::NodeIndex GetNodeIndex() const;
  const SessionState& GetSessionState() const;

  const MLValue* GetInputMLValue(int index) const;
  MLValue* GetOutputMLValue(int index);

 private:
  Status GetOrCreateOutputMLValue(int index, MLValue*& value);

  int GetInputArgIndex(int index) const;
  int GetOutputArgIndex(int index) const;

  ExecutionFrame* execution_frame_{nullptr};
  const OpKernel* kernel_{nullptr};
  const logging::Logger* logger_{nullptr};

  // The argument starting index in ExecutionFrame.
  int node_input_start_index_{-1};
  int node_output_start_index_{-1};
};

// Bring this out to a namespace to provide explicit
// specialization
template <>
struct OpKernelContext::Fetcher<Tensor, Tensor> {
  static const Tensor* Input(const OpKernelContext& ctx, int index) {
    const MLValue* p_ml_value = ctx.GetInputMLValue(index);
    return p_ml_value ? &(p_ml_value->Get<Tensor>()) : nullptr;
  }
  // Fetching output tensor without shape is not allowed except when it already exists
  static Tensor* Output(OpKernelContext& ctx, int index) {
    MLValue* p_ml_value = ctx.GetOutputMLValue(index);
    ONNXRUNTIME_ENFORCE(p_ml_value, "Please fetch output tensor with specified shape.");
    return p_ml_value->GetMutable<Tensor>();
  }
};

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

// Forward declarations for the non-specialized BuildKernel method.
template <typename T>
KernelCreateInfo BuildKernel();

namespace ml {
template <typename T>
KernelCreateInfo BuildKernel();
}  // namespace ml

namespace contrib {
template <typename T>
KernelCreateInfo BuildKernel();
}  // namespace contrib

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

#define ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMSDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

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

#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name) \
  provider##_##name##_##domain##_ver##startver##_##endver##_##type

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(name, startver, endver, type, builder, ...) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kOnnxDomain, startver, endver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(name, startver, endver, type, builder, ...) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kMLDomain, startver, endver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_MS_KERNEL(name, startver, endver, type, builder, ...) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kMSDomain, startver, endver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, ...)      \
  class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name);           \
  template <>                                                                                                      \
  KernelCreateInfo                                                                                                 \
  BuildKernel<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name)>() { \
    return KernelCreateInfo(                                                                                       \
        builder.SetName(#name)                                                                                     \
            .SetDomain(domain)                                                                                     \
            .SinceVersion(startver, endver)                                                                        \
            .Provider(provider)                                                                                    \
            .Build(),                                                                                              \
        [](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); });                              \
  }

}  // namespace onnxruntime
