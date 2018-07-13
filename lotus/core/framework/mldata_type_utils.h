#pragma once

#include "core/framework/data_types.h"
#include "core/graph/graph.h"
#include "onnx/defs/data_type_utils.h"

namespace Lotus {
namespace Utils {
MLDataType GetMLDataType(const LotusIR::NodeArg& arg);
}  // namespace Utils
}  // namespace Lotus
