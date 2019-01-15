#pragma once

#include "core/util/math_cpuonly.h"

// Eigen utils for generate proposals op
namespace onnxruntime {
namespace contrib {
template <typename T>
using ERMatXt =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ERMatXd = ERMatXt<double>;
using ERMatXf = ERMatXt<float>;
using EVecXf = Eigen::VectorXf;

// 2-d array, row major
template <typename T>
using ERArrXXt =
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using ERArrXXf = ERArrXXt<float>;

template <typename T>
using EArrXt = Eigen::Array<T, Eigen::Dynamic, 1>;

template <typename T>
using EArrXXt = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

// 1-d row vector
using ERVecXd = Eigen::RowVectorXd;
using ERVecXf = Eigen::RowVectorXf;
using EArrXf = Eigen::ArrayXf;
using EigenOuterStride = Eigen::OuterStride<Eigen::Dynamic>;
using EArrXi = Eigen::ArrayXi;
using EArrXb = EArrXt<bool>;

namespace utils {

template <typename T>
Eigen::Map<const EArrXt<T>> AsEArrXt(const std::vector<T>& arr) {
  return {arr.data(), static_cast<int>(arr.size())};
}
template <typename T>
Eigen::Map<EArrXt<T>> AsEArrXt(std::vector<T>& arr) {
  return {arr.data(), static_cast<int>(arr.size())};
}

// return a sub array of 'array' based on indices 'indices'
template <class Derived, class Derived1, class Derived2>
void GetSubArray(
    const Eigen::ArrayBase<Derived>& array,
    const Eigen::ArrayBase<Derived1>& indices,
    Eigen::ArrayBase<Derived2>* out_array) {
  ORT_ENFORCE_EQ(array.cols(), 1);
  // using T = typename Derived::Scalar;

  out_array->derived().resize(indices.size());
  for (int i = 0; i < indices.size(); i++) {
    ORT_ENFORCE_LT(indices[i], array.size());
    (*out_array)[i] = array[indices[i]];
  }
}

// return a sub array of 'array' based on indices 'indices'
template <class Derived, class Derived1>
EArrXt<typename Derived::Scalar> GetSubArray(
    const Eigen::ArrayBase<Derived>& array,
    const Eigen::ArrayBase<Derived1>& indices) {
  using T = typename Derived::Scalar;
  EArrXt<T> ret(indices.size());
  GetSubArray(array, indices, &ret);
  return ret;
}

// return a sub array of 'array' based on indices 'indices'
template <class Derived>
EArrXt<typename Derived::Scalar> GetSubArray(
    const Eigen::ArrayBase<Derived>& array,
    const std::vector<int>& indices) {
  return GetSubArray(array, AsEArrXt(indices));
}

// return 2d sub array of 'array' based on row indices 'row_indices'
template <class Derived, class Derived1, class Derived2>
void GetSubArrayRows(
    const Eigen::ArrayBase<Derived>& array2d,
    const Eigen::ArrayBase<Derived1>& row_indices,
    Eigen::ArrayBase<Derived2>* out_array) {
  out_array->derived().resize(row_indices.size(), array2d.cols());

  for (int i = 0; i < row_indices.size(); i++) {
    ORT_ENFORCE_LE(row_indices[i], array2d.size());
    out_array->row(i) =
        array2d.row(row_indices[i]).template cast<typename Derived2::Scalar>();
  }
}

// return indices of 1d array for elements evaluated to true
template <class Derived>
std::vector<int> GetArrayIndices(const Eigen::ArrayBase<Derived>& array) {
  std::vector<int> ret;
  for (int i = 0; i < array.size(); i++) {
    if (array[i]) {
      ret.push_back(i);
    }
  }
  return ret;
}
}  // namespace utils
}  // namespace contrib
}  // namespace onnxruntime
