#pragma once

#include <core/framework/ml_value.h>
#include <onnx/onnx_pb.h>

namespace Lotus {
enum class COMPARE_RESULT {
  SUCCESS,
  RESULT_DIFFERS,
  TYPE_MISMATCH,
  SHAPE_MISMATCH,
  NOT_SUPPORT
};
std::pair<COMPARE_RESULT, std::string> CompareMLValue(const MLValue& real, const MLValue& expected, const double abs_error);

//verify if the 'value' matches the 'expected' ValueInfoProto. 'value' is a model output
std::pair<COMPARE_RESULT, std::string> VerifyValueInfo(const onnx::ValueInfoProto& expected, const MLValue& value);
}  // namespace Lotus