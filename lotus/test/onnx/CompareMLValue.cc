#include "CompareMLValue.h"
#include "core/inc/op_kernel_author.h"
#include <cmath>
#include "core/inc/op_kernel_author.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/CUDA/Half.h"

using namespace Lotus;

namespace {
template <typename FLOAT_TYPE>
std::pair<EXECUTE_RESULT, size_t> CompareFloatResult(const Tensor& outvalue, const Tensor& expected_value, double abs_error) {
  const size_t size1 = expected_value.Shape().Size();
  const FLOAT_TYPE* expected_output = expected_value.Data<FLOAT_TYPE>();
  const FLOAT_TYPE* real_output = outvalue.Data<FLOAT_TYPE>();
  for (size_t di = 0; di != size1; ++di) {
    const double diff = fabs(expected_output[di] - real_output[di]);
    if (diff > abs_error) {
      return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, di);
    }
  }
  return std::make_pair(EXECUTE_RESULT::SUCCESS, -1);
}

template <typename T>
std::pair<EXECUTE_RESULT, size_t> IsResultExactlyMatch(const Tensor& outvalue, const Tensor& expected_value) {
  const size_t size1 = expected_value.Shape().Size();
  const T* expected_output = expected_value.Data<T>();
  const T* real_output = outvalue.Data<T>();
  for (size_t di = 0; di != size1; ++di) {
    if (expected_output[di] != real_output[di]) {
      return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, di);
    }
  }
  return std::make_pair(EXECUTE_RESULT::SUCCESS, -1);
}

std::pair<EXECUTE_RESULT, size_t> CompareFloat16Result(const Tensor& outvalue, const Tensor& expected_value, double abs_error) {
  const size_t size1 = expected_value.Shape().Size();
  const MLFloat16* expected_output = expected_value.Data<MLFloat16>();
  const MLFloat16* real_output = outvalue.Data<MLFloat16>();
  for (size_t di = 0; di != size1; ++di) {
    float expected = Eigen::half_impl::half_to_float(Eigen::half_impl::__half(expected_output[di].val));
    float real = Eigen::half_impl::half_to_float(Eigen::half_impl::__half(real_output[di].val));
    const double diff = fabs(expected - real);
    if (diff > abs_error) {
      return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, di);
    }
  }
  return std::make_pair(EXECUTE_RESULT::SUCCESS, -1);
}

std::pair<EXECUTE_RESULT, size_t> CompareTwoTensors(const Tensor& outvalue, const Tensor& expected_tensor, double abs_error) {
  if (expected_tensor.Shape() != outvalue.Shape()) return std::make_pair(EXECUTE_RESULT::SHAPE_MISMATCH, -1);
  auto p1 = outvalue.DataType();
  if (p1 == DataTypeImpl::GetType<float>()) {
    return CompareFloatResult<float>(outvalue, expected_tensor, abs_error);
  } else if (p1 == DataTypeImpl::GetType<double>()) {
    return CompareFloatResult<double>(outvalue, expected_tensor, abs_error);
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
    return CompareFloat16Result(outvalue, expected_tensor, abs_error);
  } else {
    return std::make_pair(EXECUTE_RESULT::NOT_SUPPORT, 0);
  }
}
template <typename T>
std::pair<EXECUTE_RESULT, size_t> CompareSeqOfMapToFloat(const T& p1, const T& p2, double abs_error) {
  if (p1.size() != p2.size()) {
    return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, 0);
  }
  for (size_t i = 0; i != p1.size(); ++i) {
    const auto& p2i = p2[i];
    if (p1[i].size() != p2i.size()) {
      return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, i);
    }

    for (const auto& pv : p1[i]) {
      auto iter = p2i.find(pv.first);
      if (iter == p2i.end()) {
        return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, i);
      }
      const double diff = fabs(iter->second - pv.second);
      if (diff > abs_error) {
        return std::make_pair(EXECUTE_RESULT::RESULT_DIFFERS, i);
      }
    }
  }
  return std::make_pair(EXECUTE_RESULT::SUCCESS, -1);
}
}  // namespace
std::pair<EXECUTE_RESULT, size_t> CompareMLValue(const Lotus::MLValue& o, const Lotus::MLValue& e, const double abs_error) {
  if (o.IsTensor() != e.IsTensor() || o.Type() != e.Type()) {
    return std::make_pair(EXECUTE_RESULT::TYPE_MISMATCH, 0);
  }
  if (!o.IsTensor()) {
    if (o.Type() == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapInt64ToFloat>(), e.Get<VectorMapInt64ToFloat>(), abs_error);
    }
    if (o.Type() == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapStringToFloat>(), e.Get<VectorMapStringToFloat>(), abs_error);
    }
    return std::make_pair(EXECUTE_RESULT::NOT_SUPPORT, 0);
  }
  const Tensor& outvalue = o.Get<Tensor>();
  const Tensor& expected_value = e.Get<Tensor>();
  if (outvalue.DataType() != expected_value.DataType()) {
    return std::make_pair(EXECUTE_RESULT::TYPE_MISMATCH, 0);
  }
  return CompareTwoTensors(outvalue, expected_value, abs_error);
}
