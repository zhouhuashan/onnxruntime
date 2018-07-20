#include "core/framework/function_kernel.h"

namespace Lotus {
    // FunctionKernel is designed to cover execution of all nodes that refers to
    // unknown <op_type>. The FunctionKernel delegates node run to corresponding
    // execution provider.
    class FunctionKernel_1_Registrar;
    template <>
    KernelCreateInfo
    BuildKernel<FunctionKernel_1_Registrar>() {
      return KernelCreateInfo(
          KernelDefBuilder().SetName(LotusIR::kFunctionOp).SinceVersion(1).Build(),
          [](const OpKernelInfo& info) -> OpKernel* { return new FunctionKernel(info); });
    };
}  // namespace Lotus

