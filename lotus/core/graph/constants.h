#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"

namespace LotusIR {
constexpr char* kNoOp = "NoOp";
constexpr char* kConstant = "Constant";
constexpr char* kFunctionOp = "_kFunctionOp";
constexpr char* kConstantValue = "value";
constexpr char* kOnnxDomain = "";
constexpr char* kMLDomain = "ai.onnx.ml";
constexpr char* kMSDomain = "com.microsoft";
constexpr char* kCpuExecutionProvider = "CPUExecutionProvider";
}  // namespace LotusIR
