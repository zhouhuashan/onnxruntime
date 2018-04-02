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

#include "core/providers/cpu/math/softmax_shared.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

void SoftmaxCPU(
    const int N,
    const int D,
    const float* Xdata,
    float* Ydata,
    float* scale,
    const float* sum_multiplier,
    bool logarithmic,
    float* rowmax) {
  Math::RowwiseMax<float, CPUMathUtil>(N, D, Xdata, rowmax, nullptr);

  // Put the intermediate result X - max(X) into Y by first copying X to Y, and then subtracting max from each entry
  // VC++ generates warning C4996 for use of std::copy with raw pointers, and I couldn't find a way to turn it off for just this line,
  // so use memcpy instead of disabling the warning everywhere.
  // std::copy(Xdata, Xdata + (N * D), Ydata);
  memcpy(Ydata, Xdata, (N * D * sizeof(*Xdata)));

  Math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasNoTrans, N, D, 1, -1, rowmax, sum_multiplier, 1, Ydata, nullptr);

  // Exponentiation
  Math::Exp<float, CPUMathUtil>(N * D, Ydata, Ydata, nullptr);
  Math::Gemv<float, CPUMathUtil>(CblasNoTrans, N, D, 1, Ydata, sum_multiplier, 0, scale, nullptr);

  // Do division
  if (!logarithmic) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        Ydata[i * D + j] /= scale[i];
      }
    }
  } else {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        Ydata[i * D + j] = Xdata[i * D + j] - rowmax[i] - log(fmaxf(scale[i], 1e-20f));
      }
    }
  }
}
}  // namespace Lotus
