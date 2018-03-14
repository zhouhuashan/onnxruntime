#ifndef CORE_FRAMEWORK_OP_KERNEL_H
#define CORE_FRAMEWORK_OP_KERNEL_H

#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/framework/execution_frame.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/op.h"

namespace Lotus {
class OpKernelContext;

class OpKernelInfo {
 public:
  explicit OpKernelInfo(const LotusIR::Node& node,
                        const AllocatorInfo& allocator_info)
      : node_(node),
        allocator_info_(allocator_info) {}

  //Get a single attribute
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  //Get repeated attributes
  template <typename T>
  Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  const LotusIR::Node& node() const {
    return node_;
  }

  const AllocatorInfo& get_allocator_info() const {
    return allocator_info_;
  }

 private:
  const LotusIR::Node& node_;
  const AllocatorInfo& allocator_info_;
};

class OpKernel {
 public:
  typedef std::function<void()> DoneCallback;

  explicit OpKernel(OpKernelInfo* info, const KernelDef* kernel_def)
      : op_kernel_info_(info),
        kernel_def_(kernel_def),
        allocator_info_(info->get_allocator_info()) {
    LOTUS_ENFORCE(nullptr != info);
  }

  const LotusIR::Node& node() const {
    return op_kernel_info_->node();
  }

  const KernelDef* kernel_def() const {
    return kernel_def_;
  }

  virtual void compute(OpKernelContext* context) = 0;
  virtual void compute_async(OpKernelContext* context, DoneCallback done) {
    UNUSED_PARAMETER(context);
    UNUSED_PARAMETER(done);
    LOTUS_NOT_IMPLEMENTED;
  }

  const AllocatorInfo& allocator() { return allocator_info_; }

 private:
  const AllocatorInfo& allocator_info_;

  OpKernelInfo* op_kernel_info_;  // TODO why is this a naked pointer? why not const ref ?

  // KernelDef of <*this> kernel, it's owned by global KernelRegistry.
  const KernelDef* kernel_def_;
};

class OpKernelContext {
 public:
  typedef std::unordered_map<std::string, size_t> ArgMap;

  explicit OpKernelContext(ExecutionFrame* frame, OpKernel* kernel)
      : execution_frame_(frame),
        kernel_(kernel) {
    LOTUS_ENFORCE(nullptr != frame && nullptr != kernel);
    arg_start_index_ = frame->get_first_arg_index(kernel->node().Index());
  }

  ~OpKernelContext(){};

  template <typename T>
  const T* input(int index) const {
    return execution_frame_->get_input<T>(arg_start_index_ + index);
  }

  // Fetch output (non-tensor) with specified index.
  template <typename T>
  T* output(int index) {
    auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->node().InputDefs().size()) + index;
    return execution_frame_->get_output<T>(output_arg_index);
  }

  // In the case that memory allocation has not been done for an output tensor,
  // The memory allocation will be done on-the-fly with given tensor shape.
  Tensor* output(int index, const TensorShape& shape);

 private:
  ExecutionFrame* execution_frame_ = nullptr;

  OpKernel* kernel_ = nullptr;

  // The argument starting index in ExecutionFrame.
  int arg_start_index_ = -1;
};

typedef OpKernel* (*KernelCreateFn)(OpKernelInfo*, const KernelDef*);

class KernelRegistry {
 public:
  // Register a kernel with kernel definition and function to create the kernel.
  Status Register(const KernelDef& kernel_def, KernelCreateFn kernel_creator) {
    auto& op_name = kernel_def.OpName();
    auto& op_domain = kernel_def.Domain();
    auto& provider = kernel_def.Provider();
    // TODO: check version overlap issue. For example, there're multiple kernels registered for same version.
    kernel_creator_fn_map_[op_name][op_domain][provider].push_back(KernelCreateInfo(kernel_def, kernel_creator));
  }

  /**/
  Status CreateKernel(const LotusIR::OperatorSchema& op_schema,
                      const ProviderType& provider_type,
                      OpKernelInfo* op_kernel_info,
                      /*out*/ OpKernel** op_kernel) const {
    // TODO: error check for op_name/op_domain/provider/since_version.
    // TODO: find the real appropriate kernel create info for specific version.
    UNUSED_PARAMETER(op_schema);
    UNUSED_PARAMETER(provider_type);
    UNUSED_PARAMETER(op_kernel_info);
    UNUSED_PARAMETER(op_kernel);
    return Status::OK();
  }

  static KernelRegistry* Instance() {
    static KernelRegistry kernel_registry;
    return &kernel_registry;
  }

 private:
  KernelRegistry() = default;

  struct KernelCreateInfo {
    KernelDef kernel_def;
    KernelCreateFn kernel_create_fn;

    KernelCreateInfo(const KernelDef& kernel_def_, KernelCreateFn kernel_create_fn_) {
      kernel_def = kernel_def_;
      kernel_create_fn = kernel_create_fn_;
    }
  };

  // Kernel create function map. Its structure is,
  // <op_name, <op_domain, <provider_type, kernel_create_functions>>>.
  std::unordered_map<std::string,
                     std::unordered_map<std::string,
                                        std::unordered_map<ProviderType, std::vector<KernelCreateInfo>>>>
      kernel_creator_fn_map_;
};
}  // namespace Lotus
#endif  // CORE_FRAMEWORK_OP_KERNEL_H
