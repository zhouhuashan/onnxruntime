#pragma once
//TODO(@chasun): move compare_mlvalue.{h,cc} to test dir

#include <core/framework/ml_value.h>
#include <string>

namespace onnx {
class ValueInfoProto;
}
namespace Lotus {
enum class COMPARE_RESULT {
  SUCCESS,
  RESULT_DIFFERS,
  TYPE_MISMATCH,
  SHAPE_MISMATCH,
  NOT_SUPPORT
};
std::pair<COMPARE_RESULT, std::string> CompareMLValue(const MLValue& real, const MLValue& expected, double per_sample_tolerance,
                                                      double relative_per_sample_tolerance, bool post_processing);

//verify if the 'value' matches the 'expected' ValueInfoProto. 'value' is a model output
std::pair<COMPARE_RESULT, std::string> VerifyValueInfo(const onnx::ValueInfoProto& expected, const MLValue& value);
}  // namespace Lotus
