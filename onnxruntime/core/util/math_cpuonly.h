/**
* Derived from caffe2, need copy right announcement here.
*/

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"

// check if we're using a GEMM implementation that does its own parallelization.
// MLAS uses Windows thread pool
// Eigen and MKL DNN will use OpenMP threads
#if defined(USE_MLAS) || defined(USE_OPENMP) || defined(USE_MKLDNN)
#define HAVE_PARALLELIZED_GEMM
#endif

namespace onnxruntime {

// common Eigen types that we will often use
template <typename T>
using EigenMatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenVectorMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
auto EigenMap(Tensor& t) { return EigenVectorMap<T>(t.MutableData<T>(), t.Shape().Size()); }
template <typename T>
auto EigenMap(const Tensor& t) { return ConstEigenVectorMap<T>(t.Data<T>(), t.Shape().Size()); }

class CPUMathUtil {
 public:
  /*CPUMathUtil contains some help method like generate a
        random seed. We only need a single instance for it.*/
  static CPUMathUtil& Instance() {
    static CPUMathUtil p;
    return p;
  }
  //todo: the random generate interface.
 private:
  CPUMathUtil() {}
};

}  // namespace onnxruntime
