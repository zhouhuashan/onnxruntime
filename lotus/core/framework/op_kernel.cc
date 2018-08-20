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
  auto output_arg_index = arg_start_index_ + static_cast<int>(kernel_->Node().InputDefs().size()) + index;
  MLValueAllocationParameters parameters;
  parameters.tensor_shape = shape;
  //@chasun: Though we don't need to give 'ret' an initial value, GCC would generate a warning if we don't do that
  //"error: 'ret' may be used uninitialized in this function"
  //This warning only exists in Release build.
  //I believe it's a false alarm.
  Tensor* ret = nullptr;
  Status status = execution_frame_->GetOrCreateMLValue<Tensor>(output_arg_index, parameters, ret);
  LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());
  return ret;
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
