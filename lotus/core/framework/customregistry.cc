#include "customregistry.h"
namespace Lotus {

CustomRegistry::CustomRegistry(bool create_func_kernel) : KernelRegistry(create_func_kernel) {}
Common::Status CustomRegistry::RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, KernelCreateFn kernel_creator) {
  return Register(kernel_def_builder, kernel_creator);
}

}  // namespace Lotus
