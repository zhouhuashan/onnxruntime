#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/graph/op.h"
#include "op_kernel_abi_wrapper.h"
#include "core/common/logging/logging.h"
using namespace ::Lotus::Common;
namespace Lotus {

Tensor* OpKernelContext::Output(int index, const TensorShape& shape) {
  if (index >= kernel_->Node().OutputDefs().size())
    return nullptr;
  // In this case, it's assumed that the tensor hasn't been allocated yet,
  // so that it's calling ExecutionFrame to create a tensor in the given position with given shape.
  MLValueAllocationParameters parameters;
  parameters.tensor_shape = shape;
  //@chasun: Though we don't need to give 'ret' an initial value, GCC would generate a warning if we don't do that
  //"error: 'ret' may be used uninitialized in this function"
  //This warning only exists in Release build.
  //I believe it's a false alarm.
  MLValue* p_ml_value = nullptr;
  Status status = execution_frame_->GetOrCreateNodeOutputMLValue(GetOutputArgIndex(index), parameters, p_ml_value);
  LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());
  return p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
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

int OpKernelContext::NumVariadicInputs(size_t arg_num) const {
  auto& arg_counts = kernel_->Node().InputArgCount();

  LOTUS_ENFORCE(arg_num < arg_counts.size(),
                "Invalid arg_num of ",
                arg_num,
                ". Num args is ",
                arg_counts.size());

  return arg_counts[arg_num];
}

Status OpKernelContext::GetTempSpaceAllocator(AllocatorPtr* output) const {
  *output = execution_frame_->GetAllocator(kernel_->Allocator(kMemTypeDefault));
  if (!*output)
    return Status(Common::LOTUS, Common::FAIL, "TempSpace allocator not found");
  return Status::OK();
}

MLDataType OpKernelContext::InputType(int index) const {
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(arg_start_index_ + index);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

MLDataType OpKernelContext::OutputType(int index) const {
  auto output_arg_index = GetOutputArgIndex(index);
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(output_arg_index);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

const MLValue* OpKernelContext::GetInputMLValue(int index) const {
  return execution_frame_->GetNodeInputOrOutputMLValue(index);
}

Fence_t OpKernelContext::InputFence(int index) const {
  if (index >= InputCount())
    return nullptr;

  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(arg_start_index_ + index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

Fence_t OpKernelContext::OutputFence(int index) const {
  if (index >= OutputCount())
    return nullptr;

  auto output_arg_index = GetOutputArgIndex(index);
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(output_arg_index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

Status OpKernelContext::GetorCreateOutputMLValue(int index, MLValue*& p_value) {
  auto output_arg_index = GetOutputArgIndex(index);
  MLValueAllocationParameters parameters;
  LOTUS_ENFORCE(execution_frame_->GetOrCreateNodeOutputMLValue(output_arg_index, parameters, p_value).IsOK());
  return Status::OK();
}

int OpKernelContext::GetOutputArgIndex(int index) const {
  return arg_start_index_ + static_cast<int>(kernel_->Node().InputDefs().size()) + index;
}

LotusIR::NodeIndex OpKernelContext::GetNodeIndex() const {
  return kernel_->Node().Index();
}

const SessionState& OpKernelContext::GetSessionState() const {
  return execution_frame_->SessionState();
}

}  // namespace Lotus
