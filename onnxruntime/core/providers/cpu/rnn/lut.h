//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: lut.h
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#pragma once

// A class of look-up tabled based mathematics.
#define USE_LUT
#include <math.h>
#include <stdint.h>

class LUT {
 public:
  ~LUT();

  bool Sigmoid2TabInit();
  bool Tanh2TabInit();

  double FastSigmoidUseLUT(double input);
  double FastTanhUseLUT(double input);

  static LUT* getInstance();

 private:
  LUT();

  double* sigmoid2tab;
  double* tanh2tab;
};

inline LUT* LUT::getInstance() {
  static LUT inst;  // The one, single instance
  return (&inst);
}

inline double LUT::FastSigmoidUseLUT(double input) {
#ifdef USE_LUT
  //PrintMatrix1D(exp2tab, 300);
  if (input > 15.0f) return 1.0f;
  if (input < -15.0f) return 0.0f;
  int64_t i = int64_t(input * 1000000) + 14999999;
  double temp = sigmoid2tab[i];
  return temp;
#else
  double exp_value;
  float return_value;

  exp_value = exp((double)-input);

  return_value = 1 / (1 + exp_value);

  return return_value;
#endif
}

inline double LUT::FastTanhUseLUT(double input) {
#ifdef USE_LUT
  // TODO: Tune the range and granularity of the lookup table.
  //PrintMatrix1D(tanh2tab, 1000);
  if (input > 15.0f) return 1.0f;
  if (input < -15.0f) return -1.0f;
  int64_t i = int64_t(input * 1000000) + 14999999;
  double temp = tanh2tab[i];
  return temp;
#else
  return tanh(input);
#endif
}
