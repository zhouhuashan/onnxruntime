#ifndef CORE_FRAMEWORK_OP_KERNEL_H
#define CORE_FRAMEWORK_OP_KERNEL_H

#include "core/common/exceptions.h"
#include "core/common/logging.h"
#include "core/common/status.h"
#include "core/framework/execution_frame.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/op.h"

namespace Lotus {
class OpKernelContext;

// A very light-weight class, which works as an aggregated
// view of all data needed for constructing a Kernel instance.
// NOTE: it does not own/hold any objects.
class OpKernelInfo {
 public:
  explicit OpKernelInfo(const LotusIR::Node& node,
                        const AllocatorInfo& allocator_info,
                        const KernelDef& kernel_def)
      : node_(node),
        allocator_info_(allocator_info),
        kernel_def_(kernel_def) {}

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

  const KernelDef& get_kernel_def() const {
    return kernel_def_;
  }

 private:
  const LotusIR::Node& node_;
  const AllocatorInfo& allocator_info_;
  const KernelDef& kernel_def_;
};

class OpKernel {
 public:
  typedef std::function<void()> DoneCallback;

  explicit OpKernel(const OpKernelInfo& info)
      : op_kernel_info_(info) {
    // TODO: enable this
    // LOTUS_ENFORCE(nullptr != kernel_def, "kernel_def should be not nullptr.")
  }

  const LotusIR::Node& node() const {
    return op_kernel_info_.node();
  }

  const KernelDef& kernel_def() const {
    return op_kernel_info_.get_kernel_def();
  }

  virtual void compute(OpKernelContext* context) = 0;
  virtual void compute_async(OpKernelContext* context, DoneCallback done) {
    UNUSED_PARAMETER(context);
    UNUSED_PARAMETER(done);
    LOTUS_NOT_IMPLEMENTED;
  }

  const AllocatorInfo& allocator() { return op_kernel_info_.get_allocator_info(); }

 protected:
  OpKernelInfo op_kernel_info_;

  const KernelDef* kernel_def_;
};

class OpKernelContext {
 public:
  typedef std::unordered_map<std::string, size_t> ArgMap;

  explicit OpKernelContext(ExecutionFrame* frame, OpKernel* kernel);

  ~OpKernelContext(){};

  template <typename T>
  const T* input(int index) const {
    return execution_frame_->get_value<T>(arg_start_index_ + index);
  }

  // Fetch output (non-tensor) with specified index.
  template <typename T>
  T* output(int index) {
    auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->node().InputDefs().size()) + index;
    return execution_frame_->get_mutable_value<T>(output_arg_index);
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

typedef OpKernel* (*KernelCreateFn)(const OpKernelInfo&);

class KernelRegistry {
 public:
  // Register a kernel with kernel definition and function to create the kernel.
  Status Register(const KernelDef& kernel_def, KernelCreateFn kernel_creator) {
    auto& op_name = kernel_def.OpName();
    auto& op_domain = kernel_def.Domain();
    auto& provider = kernel_def.Provider();
    // TODO: check version overlap issue. For example, there're multiple kernels registered for same version.
    kernel_creator_fn_map_[op_name][op_domain][provider].push_back(KernelCreateInfo(kernel_def, kernel_creator));
    return Status::OK();
  }

  /**/
  // factory functions should always return a unique_ptr for maximum flexibility
  // for its clients unless the factory is managing the lifecycle of the pointer
  // itself.
  Status CreateKernel(const LotusIR::OperatorSchema& op_schema,
                      const ProviderType& provider_type,
                      const LotusIR::Node& node,
                      const AllocatorInfo& allocator_info,
                      /*out*/ std::unique_ptr<OpKernel>* op_kernel) const {
    // TODO: error check for op_name/op_domain/provider/since_version.
    // TODO: find the real appropriate kernel create info for specific version.
    UNUSED_PARAMETER(op_schema);
    UNUSED_PARAMETER(provider_type);
    UNUSED_PARAMETER(node);
    UNUSED_PARAMETER(allocator_info);
    UNUSED_PARAMETER(op_kernel);

    // TODO following code exists for testing only
    // Please replace it with real code
    auto& name = op_schema.GetName();
    auto& domain = op_schema.Domain();
    auto it = kernel_creator_fn_map_.find(name);
    if (it == kernel_creator_fn_map_.end()) {
      LOG(ERROR) << "Could not find op name: " << name;
      return Status(LOTUS, FAIL, "Kernel not found");
    }
    auto it2 = it->second.find(domain);
    if (it2 == it->second.end()) {
      LOG(ERROR) << "Could not find op domain: " << domain;
      return Status(LOTUS, FAIL, "Kernel not found");
    }
    auto it3 = it2->second.find(provider_type);
    if (it3 == it2->second.end() || it3->second.empty()) {
      LOG(ERROR) << "Could not find provider_type: " << provider_type;
      return Status(LOTUS, FAIL, "Kernel not found");
    }
    // pick the first one
    auto fn = it3->second.front().kernel_create_fn;
    OpKernelInfo info(node, allocator_info, it3->second.front().kernel_def);
    op_kernel->reset(fn(info));

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

#define REGISTER_KERNEL(kernel_def, ...) REGISTER_KERNEL_UNIQ_HELPER(__COUNTER__, kernel_def, __VA_ARGS__)
#define REGISTER_KERNEL_UNIQ_HELPER(counter, kernel_def, ...) REGISTER_KERNEL_UNIQ(counter, kernel_def, __VA_ARGS__)
#define REGISTER_KERNEL_UNIQ(counter, kernel_def, ...)                                                          \
  static Lotus::Common::Status kernel_def_##counter##_status = KernelRegistry::Instance()->Register(kernel_def, \
                                                                                                    [](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); });

}  // namespace Lotus
#endif  // CORE_FRAMEWORK_OP_KERNEL_H
