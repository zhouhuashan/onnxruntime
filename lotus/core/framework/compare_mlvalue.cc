#include "compare_mlvalue.h"
#include <cmath>
#include <sstream>

#include "core/inc/op_kernel_author.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/CUDA/Half.h"

using namespace Lotus;

namespace {
template <typename FLOAT_TYPE>
std::pair<COMPARE_RESULT, std::string> CompareFloatResult(const Tensor& outvalue, const Tensor& expected_value, double per_sample_tolerance, double relative_per_sample_tolerance) {
  const size_t size1 = expected_value.Shape().Size();
  const FLOAT_TYPE* expected_output = expected_value.Data<FLOAT_TYPE>();
  const FLOAT_TYPE* real_output = outvalue.Data<FLOAT_TYPE>();
  for (size_t di = 0; di != size1; ++di) {
    const double diff = fabs(expected_output[di] - real_output[di]);
    const double rtol = per_sample_tolerance + relative_per_sample_tolerance * abs(expected_output[di]);
    if (diff > rtol) {
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

template <typename T>
std::pair<COMPARE_RESULT, std::string> IsResultExactlyMatch(const Tensor& outvalue, const Tensor& expected_value) {
  const size_t size1 = expected_value.Shape().Size();
  const T* expected_output = expected_value.Data<T>();
  const T* real_output = outvalue.Data<T>();
  for (size_t di = 0; di != size1; ++di) {
    if (expected_output[di] != real_output[di]) {
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

std::pair<COMPARE_RESULT, std::string> CompareFloat16Result(const Tensor& outvalue, const Tensor& expected_value, double per_sample_tolerance, double relative_per_sample_tolerance) {
  const size_t size1 = expected_value.Shape().Size();
  const MLFloat16* expected_output = expected_value.Data<MLFloat16>();
  const MLFloat16* real_output = outvalue.Data<MLFloat16>();
  for (size_t di = 0; di != size1; ++di) {
    float expected = Eigen::half_impl::half_to_float(Eigen::half_impl::__half(expected_output[di].val));
    float real = Eigen::half_impl::half_to_float(Eigen::half_impl::__half(real_output[di].val));
    const double diff = fabs(expected - real);
    const double rtol = per_sample_tolerance + relative_per_sample_tolerance * abs(expected);
    if (diff > rtol) {
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

std::pair<COMPARE_RESULT, std::string> CompareTwoTensors(const Tensor& outvalue, const Tensor& expected_tensor, double per_sample_tolerance, double relative_per_sample_tolerance) {
  if (expected_tensor.Shape() != outvalue.Shape()) {
    std::ostringstream oss;
    oss << "shape mismatch, expect " << expected_tensor.Shape().ToString() << " got " << outvalue.Shape().ToString();
    return std::make_pair(COMPARE_RESULT::SHAPE_MISMATCH, oss.str());
  }
  auto p1 = outvalue.DataType();
  if (p1 == DataTypeImpl::GetType<float>()) {
    return CompareFloatResult<float>(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance);
  } else if (p1 == DataTypeImpl::GetType<double>()) {
    return CompareFloatResult<double>(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance);
  } else if (p1 == DataTypeImpl::GetType<std::string>()) {
    return IsResultExactlyMatch<std::string>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<uint8_t>()) {
    return IsResultExactlyMatch<uint8_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int8_t>()) {
    return IsResultExactlyMatch<int8_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<uint16_t>()) {
    return IsResultExactlyMatch<uint16_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int16_t>()) {
    return IsResultExactlyMatch<int16_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<uint32_t>()) {
    return IsResultExactlyMatch<uint32_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int32_t>()) {
    return IsResultExactlyMatch<int32_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<uint64_t>()) {
    return IsResultExactlyMatch<uint64_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int64_t>()) {
    return IsResultExactlyMatch<int64_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<bool>()) {
    return IsResultExactlyMatch<bool>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<MLFloat16>()) {
    return CompareFloat16Result(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance);
  } else {
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "");
  }
}
template <typename T>
std::pair<COMPARE_RESULT, std::string> CompareSeqOfMapToFloat(const T& p1, const T& expected_value, double per_sample_tolerance, double relative_per_sample_tolerance) {
  if (p1.size() != expected_value.size()) {
    return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
  }
  for (size_t i = 0; i != p1.size(); ++i) {
    const auto& p2i = expected_value[i];
    if (p1[i].size() != p2i.size()) {
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
    }

    for (const auto& pv : p1[i]) {
      auto iter = p2i.find(pv.first);
      if (iter == p2i.end()) {
        return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
      }
      const double diff = fabs(iter->second - pv.second);
      const double rtol = per_sample_tolerance + relative_per_sample_tolerance * abs(iter->second);
      if (diff > rtol) {
        return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
      }
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

const char* ElementTypeToString(MLDataType type) {
  if (type == DataTypeImpl::GetType<float>()) {
    return "tensor(float)";
  } else if (type == DataTypeImpl::GetType<bool>()) {
    return "tensor(bool)";
  }

  else if (type == DataTypeImpl::GetType<int32_t>()) {
    return "tensor(int32)";
  }

  else if (type == DataTypeImpl::GetType<double>()) {
    return "tensor(double)";
  }

  else if (type == DataTypeImpl::GetType<std::string>()) {
    return "tensor(string)";
  }

  else if (type == DataTypeImpl::GetType<uint8_t>()) {
    return "tensor(uint8)";
  }

  else if (type == DataTypeImpl::GetType<uint16_t>()) {
    return "tensor(uint16)";
  }

  else if (type == DataTypeImpl::GetType<int16_t>()) {
    return "tensor(int16)";
  }

  else if (type == DataTypeImpl::GetType<int64_t>()) {
    return "tensor(int64)";
  }

  else if (type == DataTypeImpl::GetType<uint32_t>()) {
    return "tensor(uint32)";
  }

  else if (type == DataTypeImpl::GetType<uint64_t>()) {
    return "tensor(uint64)";
  }

  else if (type == DataTypeImpl::GetType<MLFloat16>()) {
    return "tensor(MLFloat16)";
  } else {
    return "unknown";
  }
}

bool is_shape_equal(const TensorShape& v1, const ::onnx::TensorShapeProto& v2) {
  const int len = v2.dim_size();
  //because v1.NumDimensions() cannot be negative
  if (len < 0) return false;
  if (v1.NumDimensions() != static_cast<size_t>(len)) return false;
  for (int i = 0; i != len; ++i) {
    ::google::protobuf::int64 d = v2.dim(i).dim_value();
    //dim value can be zero or negative, in such case, we assume it can match any value
    if (d > 0 && d != v1[i]) return false;
  }
  return true;
}

std::string ToString(const ::onnx::TensorShapeProto& shape) {
  std::string result;

  result.append("{");
  bool first = true;
  for (int i = 0; i != shape.dim_size(); i++) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(shape.dim(i).dim_value()));
    first = false;
  }
  result.append("}");

  return result;
}
}  // namespace

namespace Lotus {
std::pair<COMPARE_RESULT, std::string> CompareMLValue(const MLValue& o, const MLValue& expected_mlvalue, double per_sample_tolerance, double relative_per_sample_tolerance) {
  if (o.IsTensor() != expected_mlvalue.IsTensor() || o.Type() != expected_mlvalue.Type()) {
    return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
  }
  if (!o.IsTensor()) {
    if (o.Type() == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapInt64ToFloat>(), expected_mlvalue.Get<VectorMapInt64ToFloat>(), per_sample_tolerance, relative_per_sample_tolerance);
    }
    if (o.Type() == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapStringToFloat>(), expected_mlvalue.Get<VectorMapStringToFloat>(), per_sample_tolerance, relative_per_sample_tolerance);
    }
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "");
  }
  const Tensor& outvalue = o.Get<Tensor>();
  const Tensor& expected_tensor = expected_mlvalue.Get<Tensor>();
  if (outvalue.DataType() != expected_tensor.DataType()) {
    std::ostringstream oss;
    oss << "expect " << ElementTypeToString(expected_tensor.DataType()) << " got " << ElementTypeToString(outvalue.DataType());
    return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, oss.str());
  }
  return CompareTwoTensors(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance);
}

std::pair<COMPARE_RESULT, std::string> VerifyValueInfo(const onnx::ValueInfoProto& v, const MLValue& o) {
  if (v.has_type()) {
    if (v.type().has_tensor_type()) {
      if (o.Type() != DataTypeImpl::GetType<Tensor>()) {
        return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
      }
      ::onnx::TypeProto_Tensor t = v.type().tensor_type();
      //below code doesn't work
      //if (((TensorTypeBase*)o.Type())->GetElementType() != DataTypeImpl::ElementTypeFromProto(t.elem_type())) {
      //	return COMPARE_RESULT::TYPE_MISMATCH;
      //}
      const Tensor& o1 = o.Get<Tensor>();
      if (o1.DataType() != DataTypeImpl::ElementTypeFromProto(t.elem_type())) {
        return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
      }
      if (!is_shape_equal(o1.Shape(), t.shape())) {
        std::ostringstream oss;
        oss << "Tensor shape mismatch, model file expects " << ToString(t.shape()) << ", real output is " << o1.Shape().ToString();
        return std::make_pair(COMPARE_RESULT::SHAPE_MISMATCH, oss.str());
      }
    } else {
      //Cannot do this check for tensor type.
      //For tensor type, o.Type() is TensorTypeBase*, but p points to a subclass of TensorTypeBase
      auto p = DataTypeImpl::TypeFromProto(v.type());
      if (o.Type() != p) {
        return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
      }
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}
}  // namespace Lotus
