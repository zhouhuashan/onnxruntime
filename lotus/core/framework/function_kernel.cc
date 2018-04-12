#include "core/framework/function_kernel.h"

namespace Lotus {
    // FunctionKernel is designed to cover execution of all nodes that refers to
    // unknown <op_type>. The FunctionKernel delegates node run to corresponding
    // execution provider.
    REGISTER_KERNEL(KernelDefBuilder(LotusIR::kFunctionOp).SinceVersion(1),
        FunctionKernel);
}  // namespace Lotus
