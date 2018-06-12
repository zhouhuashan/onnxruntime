#include "CompareMLValue.h"
#include "core/inc/op_kernel_author.h"
#include <cmath>
#include "core/inc/op_kernel_author.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/CUDA/Half.h"

using namespace Lotus;

namespace {
template <typename FLOAT_TYPE>
std::pair<EXECUTE_RESULT, size_t> compare_float_result(const Tensor& outvalue, const Tensor& expected_value, const FLOAT_TYPE abs_error) {
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
std::pair<EXECUTE_RESULT, size_t> is_result_exactly_match(const Tensor& outvalue, const Tensor& expected_value) {
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

std::pair<EXECUTE_RESULT, size_t> compare_float16_result(const Tensor& outvalue, const Tensor& expected_value, const float abs_error) {
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

std::pair<EXECUTE_RESULT, size_t> compare(const Tensor& outvalue, const Tensor& expected_tensor) {
  if (expected_tensor.Shape() != outvalue.Shape()) return std::make_pair(EXECUTE_RESULT::SHAPE_MISMATCH, -1);
  auto p1 = outvalue.DataType();
  if (p1 == DataTypeImpl::GetType<float>()) {
    return compare_float_result<float>(outvalue, expected_tensor, 1e-5f);
  } else if (p1 == DataTypeImpl::GetType<double>()) {
    return compare_float_result<double>(outvalue, expected_tensor, 1e-5);
  } else if (p1 == DataTypeImpl::GetType<uint8_t>()) {
    return is_result_exactly_match<uint8_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int8_t>()) {
    return is_result_exactly_match<int8_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<uint16_t>()) {
    return is_result_exactly_match<uint16_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int16_t>()) {
    return is_result_exactly_match<int16_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<uint32_t>()) {
    return is_result_exactly_match<uint32_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int32_t>()) {
    return is_result_exactly_match<int32_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<uint64_t>()) {
    return is_result_exactly_match<uint64_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<int64_t>()) {
    return is_result_exactly_match<int64_t>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<bool>()) {
    return is_result_exactly_match<bool>(outvalue, expected_tensor);
  } else if (p1 == DataTypeImpl::GetType<MLFloat16>()) {
    return compare_float16_result(outvalue, expected_tensor, 1e-3f);
  } else {
    return std::make_pair(EXECUTE_RESULT::NOT_SUPPORT, 0);
  }
}
}  // namespace
std::pair<EXECUTE_RESULT, size_t> compareMLValue(const Lotus::MLValue& o, const Lotus::MLValue& e) {
  if (o.IsTensor() != e.IsTensor()) {
    return std::make_pair(EXECUTE_RESULT::TYPE_MISMATCH, 0);
  }
  if (!o.IsTensor()) {
    return std::make_pair(EXECUTE_RESULT::NOT_SUPPORT, 0);
  }
  const Tensor& outvalue = o.Get<Tensor>();
  const Tensor& expected_value = e.Get<Tensor>();
  if (outvalue.DataType() != expected_value.DataType()) {
    return std::make_pair(EXECUTE_RESULT::TYPE_MISMATCH, 0);
  }
  return compare(outvalue, expected_value);
}
