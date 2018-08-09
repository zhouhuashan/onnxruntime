#include "customregistry.h"
namespace Lotus {

Common::Status CustomRegistry::RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, KernelCreateFn kernel_creator) {
  return Register(kernel_def_builder, kernel_creator);
}

}  // namespace Lotus
