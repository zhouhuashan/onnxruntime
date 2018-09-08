#pragma once
#include <tvm/tvm.h>

#include "core/framework/data_types.h"

namespace Lotus {

constexpr const char* TVM_STACKVM = "TvmStackVm";

namespace tvm_codegen {
  // Helper function that converts a Lotus MLDataType to TVM DLDataType
  DLDataType ToTvmDLDataType(MLDataType ml_type);
}  //  namespace tvm
}  //  namespace Lotus
