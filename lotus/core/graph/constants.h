#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"

namespace LotusIR {
static const std::string kNoOp = "NoOp";
static const std::string kConstant = "Constant";
static const std::string kConstantValue = "value";
static const std::string kOnnxDomain = "";
static const std::string kMLDomain = "ai.onnx.ml";
static const std::string kMSDomain = "com.microsoft";
static const std::string kCpuExecutionProvider = "CPUExecutionProvider";

}  // namespace LotusIR
