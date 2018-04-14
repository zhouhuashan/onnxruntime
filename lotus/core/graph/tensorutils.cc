#include "core/graph/tensorutils.h"

#include <algorithm>
#include "gsl/span"

namespace Lotus {
namespace Utils {
Status TensorUtils::UnpackTensor(const onnx::TensorProto& tensor, /*out*/ std::string* p_data, int64_t expected_size) {
  if (onnx::TensorProto_DataType_STRING != tensor.data_type() || nullptr == p_data) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }

  if (tensor.string_data_size() != expected_size)
    return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  const auto data = gsl::make_span(p_data, expected_size);

  auto& string_data = tensor.string_data();
  std::copy(string_data.cbegin(), string_data.cend(), data.begin());

  return Status::OK();
}

Status TensorUtils::UnpackTensor(const onnx::TensorProto& tensor, /*out*/ bool* p_data, int64_t expected_size) {
  if (onnx::TensorProto_DataType_BOOL != tensor.data_type() || nullptr == p_data) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(bool))
      return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor, p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  const auto data = gsl::make_span(p_data, expected_size);
  std::copy(tensor.int32_data().cbegin(), tensor.int32_data().cend(), data.begin());

  return Status::OK();
}
}  // namespace Utils
}  // namespace Lotus
