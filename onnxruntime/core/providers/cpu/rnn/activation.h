//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: activation.h
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#pragma once
#include <immintrin.h>
#include <math.h>

#define TANH tanh
#define SIGMOID Sigmoid_Hard

#define TANH5 fast_tanh_contfrac5
#define SIGMOID5 fast_sigmoid_contfrac5

#define TANH7 fast_tanh_contfrac7
#define SIGMOID7 fast_sigmoid_contfrac7

#define __Init_Contfrac_Avx_Constants__ \
  __m256 c15 = _mm256_set1_ps(15.0f);   \
  __m256 c105 = _mm256_set1_ps(105.0f); \
  __m256 c420 = _mm256_set1_ps(420.0f); \
  __m256 c945 = _mm256_set1_ps(945.0f); \
  __m256 chalf = _mm256_set1_ps(0.5f);  \
  __m256 c1 = _mm256_set1_ps(1.0f);     \
  __m256 c1fifth = _mm256_set1_ps(0.2f)

#define __Init_Contfrac7_Avx_Constants__      \
  __m256 c28 = _mm256_set1_ps(28.0f);         \
  __m256 c378 = _mm256_set1_ps(378.0f);       \
  __m256 c3150 = _mm256_set1_ps(3150.0f);     \
  __m256 c17325 = _mm256_set1_ps(17325.0f);   \
  __m256 c62370 = _mm256_set1_ps(62370.0f);   \
  __m256 c135135 = _mm256_set1_ps(135135.0f); \
  __m256 chalf = _mm256_set1_ps(0.5f);        \
  __m256 c1 = _mm256_set1_ps(1.0f);           \
  __m256 c1fifth = _mm256_set1_ps(0.2f)

#define __Init_Contfrac5_Avx_Bounds__ \
  __m256 ub = _mm256_set1_ps(3.65f);  \
  __m256 lb = _mm256_set1_ps(-3.65f); \
  __m256 xsq, xnum

#define __Init_Contfrac7_Avx_Bounds__ \
  __m256 ub = _mm256_set1_ps(5.0f);   \
  __m256 lb = _mm256_set1_ps(-5.0f);  \
  __m256 xsq, xnum

#define __Init_Hard_Sigmoid_Avx_Bounds__ \
  __m256 hsigub = _mm256_set1_ps(2.5f);  \
  __m256 hsiglb = _mm256_set1_ps(-2.5f)

#define __Fast_Sigmoid_Contfrac5_Avx__(x1, x3) \
  x1 = _mm256_mul_ps(x1, chalf);               \
  x1 = _mm256_min_ps(x1, ub);                  \
  x1 = _mm256_max_ps(x1, lb);                  \
  xsq = _mm256_mul_ps(x1, x1);                 \
  xnum = _mm256_add_ps(xsq, c105);             \
  xnum = _mm256_mul_ps(xnum, xsq);             \
  xnum = _mm256_add_ps(xnum, c945);            \
  xnum = _mm256_mul_ps(x1, xnum);              \
  x3 = _mm256_mul_ps(xsq, c15);                \
  x3 = _mm256_add_ps(x3, c420);                \
  x3 = _mm256_mul_ps(x3, xsq);                 \
  x3 = _mm256_add_ps(x3, c945);                \
  x3 = _mm256_div_ps(xnum, x3);                \
  x3 = _mm256_add_ps(x3, c1);                  \
  x3 = _mm256_mul_ps(x3, chalf)

#define __Fast_Tanh_Contfrac5_Avx__(x1, x3) \
  x1 = _mm256_min_ps(x1, ub);               \
  x1 = _mm256_max_ps(x1, lb);               \
  xsq = _mm256_mul_ps(x1, x1);              \
  xnum = _mm256_add_ps(xsq, c105);          \
  xnum = _mm256_mul_ps(xnum, xsq);          \
  xnum = _mm256_add_ps(xnum, c945);         \
  xnum = _mm256_mul_ps(x1, xnum);           \
  x3 = _mm256_mul_ps(xsq, c15);             \
  x3 = _mm256_add_ps(x3, c420);             \
  x3 = _mm256_mul_ps(x3, xsq);              \
  x3 = _mm256_add_ps(x3, c945);             \
  x3 = _mm256_div_ps(xnum, x3)

#define __Fast_Sigmoid_Contfrac7_Avx__(x1, x3) \
  x1 = _mm256_mul_ps(x1, chalf);               \
  x1 = _mm256_min_ps(x1, ub);                  \
  x1 = _mm256_max_ps(x1, lb);                  \
  xsq = _mm256_mul_ps(x1, x1);                 \
  xnum = _mm256_add_ps(xsq, c378);             \
  xnum = _mm256_mul_ps(xnum, xsq);             \
  xnum = _mm256_add_ps(xnum, c17325);          \
  xnum = _mm256_mul_ps(xnum, xsq);             \
  xnum = _mm256_add_ps(xnum, c135135);         \
  xnum = _mm256_mul_ps(x1, xnum);              \
  x3 = _mm256_mul_ps(xsq, c28);                \
  x3 = _mm256_add_ps(x3, c3150);               \
  x3 = _mm256_mul_ps(x3, xsq);                 \
  x3 = _mm256_add_ps(x3, c62370);              \
  x3 = _mm256_mul_ps(x3, xsq);                 \
  x3 = _mm256_add_ps(x3, c135135);             \
  x3 = _mm256_div_ps(xnum, x3);                \
  x3 = _mm256_add_ps(x3, c1);                  \
  x3 = _mm256_mul_ps(x3, chalf)

#define __Fast_Tanh_Contfrac7_Avx__(x1, x3) \
  x1 = _mm256_min_ps(x1, ub);               \
  x1 = _mm256_max_ps(x1, lb);               \
  xsq = _mm256_mul_ps(x1, x1);              \
  xnum = _mm256_add_ps(xsq, c378);          \
  xnum = _mm256_mul_ps(xnum, xsq);          \
  xnum = _mm256_add_ps(xnum, c17325);       \
  xnum = _mm256_mul_ps(xnum, xsq);          \
  xnum = _mm256_add_ps(xnum, c135135);      \
  xnum = _mm256_mul_ps(x1, xnum);           \
  x3 = _mm256_mul_ps(xsq, c28);             \
  x3 = _mm256_add_ps(x3, c3150);            \
  x3 = _mm256_mul_ps(x3, xsq);              \
  x3 = _mm256_add_ps(x3, c62370);           \
  x3 = _mm256_mul_ps(x3, xsq);              \
  x3 = _mm256_add_ps(x3, c135135);          \
  x3 = _mm256_div_ps(xnum, x3);

#define __Hard_Sigmoid_Avx__(x1, x3) \
  x1 = _mm256_min_ps(x1, hsigub);    \
  x1 = _mm256_max_ps(x1, hsiglb);    \
  x3 = _mm256_add_ps(_mm256_mul_ps(x1, c1fifth), chalf)

#define __Init_ReLu_Avx__ \
  __m256 zero = _mm256_set1_ps(0.0f)

#define __ReLu__Avx(x, y) \
  y = _mm256_max_ps(x, zero)

inline float fast_tanh_contfrac5(float x) {
  if (x >= 3.65f) return 1.0f;
  if (x <= -3.65f) return -1.0f;
  float x2 = x * x;
  float a = x * (945.0f + x2 * (105.0f + x2));
  float b = 945.0f + x2 * (420.0f + x2 * 15.0f);
  return a / b;
}

inline void fast_tanh_confrac5_avx(const float* ps1, const float* ps2, const float* ps3, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;
  __m256 c15 = _mm256_set1_ps(15.0f);
  __m256 c105 = _mm256_set1_ps(105.0f);
  __m256 c420 = _mm256_set1_ps(420.0f);
  __m256 c945 = _mm256_set1_ps(945.0f);
  __m256 ub = _mm256_set1_ps(5.0f);
  __m256 lb = _mm256_set1_ps(-5.0f);

  for (; pd < pdLim8; ps1 += 8, ps2 += 8, ps3 += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps1);
    __m256 x2 = _mm256_loadu_ps(ps2);
    __m256 x3 = _mm256_loadu_ps(ps3);
    x1 = _mm256_add_ps(x1, x2);
    x1 = _mm256_add_ps(x1, x3);
    x1 = _mm256_min_ps(x1, ub);
    x1 = _mm256_max_ps(x1, lb);
    x2 = _mm256_mul_ps(x1, x1);

    //float a = x * (945.0f + x2 * (105.0f + x2));
    //float b = 945.0f + x2 * (420.0f + x2 * 15.0f);
    __m256 x4 = _mm256_add_ps(x2, c105);
    x4 = _mm256_mul_ps(x4, x2);
    x4 = _mm256_add_ps(x4, c945);
    x4 = _mm256_mul_ps(x1, x4);

    x3 = _mm256_mul_ps(x2, c15);
    x3 = _mm256_add_ps(x3, c420);
    x3 = _mm256_mul_ps(x3, x2);
    x3 = _mm256_add_ps(x3, c945);

    x3 = _mm256_div_ps(x4, x3);

    _mm256_storeu_ps(pd, x3);
  }

  for (; pd < pdLim; ps1++, ps2++, ps3++, pd++) {
    *pd = fast_tanh_contfrac5((*ps1) + (*ps2) + (*ps3));
  }
}

inline void fast_tanh_confrac5_avx(const float* ps1, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;
  __m256 c15 = _mm256_set1_ps(15.0f);
  __m256 c105 = _mm256_set1_ps(105.0f);
  __m256 c420 = _mm256_set1_ps(420.0f);
  __m256 c945 = _mm256_set1_ps(945.0f);
  __m256 ub = _mm256_set1_ps(5.0f);
  __m256 lb = _mm256_set1_ps(-5.0f);

  for (; pd < pdLim8; ps1 += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps1);
    x1 = _mm256_min_ps(x1, ub);
    x1 = _mm256_max_ps(x1, lb);
    __m256 x2 = _mm256_mul_ps(x1, x1);

    //float a = x * (945.0f + x2 * (105.0f + x2));
    //float b = 945.0f + x2 * (420.0f + x2 * 15.0f);
    __m256 x4 = _mm256_add_ps(x2, c105);
    x4 = _mm256_mul_ps(x4, x2);
    x4 = _mm256_add_ps(x4, c945);
    x4 = _mm256_mul_ps(x1, x4);

    __m256 x3 = _mm256_mul_ps(x2, c15);
    x3 = _mm256_add_ps(x3, c420);
    x3 = _mm256_mul_ps(x3, x2);
    x3 = _mm256_add_ps(x3, c945);

    x3 = _mm256_div_ps(x4, x3);

    _mm256_storeu_ps(pd, x3);
  }

  for (; pd < pdLim; ps1++, pd++) {
    *pd = fast_tanh_contfrac5((*ps1));
  }
}

inline float fast_sigmoid_contfrac5(float x) {
  if (x >= 7.31f) return 1.0f;
  if (x <= -7.31f) return 0.0f;
  x = 0.5f * x;
  float x2 = x * x;
  float a = x * (945.0f + x2 * (105.0f + x2));
  float b = 945.0f + x2 * (420.0f + x2 * 15.0f);
  return 0.5f * (a / b + 1);
}

inline void fast_sigmoid_contfrac5_avx(const float* ps, float* pd, int c) {
  //float* ps0 = (float*)ps;
  //float * pd0 = pd;
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (; pd < pdLim8; ps += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps);
    __m256 x3;
    __Fast_Sigmoid_Contfrac5_Avx__(x1, x3);
    _mm256_storeu_ps(pd, x3);
  }

  for (; pd < pdLim; ps++, pd++) {
    *pd = fast_sigmoid_contfrac5((*ps));
  }
}

inline void fast_sigmoid_contfrac5_avx(const float* ps, float* pd, int c, float bias) {
  //float* ps0 = (float*)ps;
  //float* pd0 = pd;
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;
  __m256 b = _mm256_set1_ps(bias);

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (; pd < pdLim8; ps += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps);
    x1 = _mm256_add_ps(x1, b);
    __m256 x3;
    __Fast_Sigmoid_Contfrac5_Avx__(x1, x3);
    _mm256_storeu_ps(pd, x3);
  }

  for (; pd < pdLim; ps++, pd++) {
    *pd = fast_sigmoid_contfrac5((*ps) + bias);
  }
}

inline void reset_gate_contfrac5_avx(const float* ps1, const float* ps2, const float* ps3, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (; pd < pdLim8; ps1 += 8, ps2 += 8, ps3 += 8, ps4 += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps1);
    __m256 x2 = _mm256_loadu_ps(ps2);
    __m256 x3 = _mm256_loadu_ps(ps3);
    __m256 s_prev = _mm256_loadu_ps(ps4);
    x1 = _mm256_add_ps(x1, x2);
    x1 = _mm256_add_ps(x1, x3);
    __Fast_Sigmoid_Contfrac5_Avx__(x1, x3);
    x3 = _mm256_mul_ps(x3, s_prev);
    _mm256_storeu_ps(pd, x3);
  }

  for (; pd < pdLim;) {
    *(pd++) = (*(ps4++)) * fast_sigmoid_contfrac5((*(ps1++)) + (*(ps2++)) + (*(ps3++)));
  }
}

inline void output_gate_contfrac5_avx(const float* ps11, const float* ps12, const float* ps13, const float* ps21, const float* ps22, const float* ps23, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (; pd < pdLim8; ps11 += 8, ps12 += 8, ps13 += 8, ps21 += 8, ps22 += 8, ps23 += 8, ps4 += 8, pd += 8) {
    __m256 x11 = _mm256_loadu_ps(ps11);
    __m256 x12 = _mm256_loadu_ps(ps12);
    __m256 x13 = _mm256_loadu_ps(ps13);
    __m256 x21 = _mm256_loadu_ps(ps21);
    __m256 x22 = _mm256_loadu_ps(ps22);
    __m256 x23 = _mm256_loadu_ps(ps23);
    __m256 s_prev = _mm256_loadu_ps(ps4);

    x11 = _mm256_add_ps(x11, x12);
    x11 = _mm256_add_ps(x11, x13);
    x21 = _mm256_add_ps(x21, x22);
    x21 = _mm256_add_ps(x21, x23);

    __Fast_Sigmoid_Contfrac5_Avx__(x11, x13);
    __Fast_Tanh_Contfrac5_Avx__(x21, x23);

    __m256 x14 = _mm256_mul_ps(x13, s_prev);
    x13 = _mm256_sub_ps(c1, x13);
    x23 = _mm256_mul_ps(x13, x23);
    x23 = _mm256_add_ps(x23, x14);

    _mm256_storeu_ps(pd, x23);
  }

  for (; pd < pdLim;) {
    float tmp_z = fast_sigmoid_contfrac5((*(ps11++)) + (*(ps12++)) + (*(ps13++)));
    float tmp_h = fast_tanh_contfrac5((*(ps21++)) + (*(ps22++)) + (*(ps23++)));
    *(pd++) = (1 - tmp_z) * tmp_h + tmp_z * (*(ps4++));
  }
}

inline float fast_tanh_contfrac6(float x) {
  if (x >= 5.0f) return 1.0f;
  if (x <= -5.0f) return -1.0f;
  float x2 = x * x;
  float a = x * (10395.0f + x2 * (1260.0f + x2 * 21.0f));
  float b = 10395.0f + x2 * (4725.0f + x2 * (210.0f + x2));
  return a / b;
}

inline float fast_sigmoid_contfrac6(float x) {
  if (x >= 10.0f) return 1.0f;
  if (x <= -10.0f) return -0.0f;
  x = 0.5f * x;
  float x2 = x * x;
  float a = x * (10395.0f + x2 * (1260.0f + x2 * 21.0f));
  float b = 10395.0f + x2 * (4725.0f + x2 * (210.0f + x2));
  return 0.5f * (a / b + 1);
}

inline float fast_tanh_contfrac7(float x) {
  if (x >= 5.0f) return 1.0f;
  if (x <= -5.0f) return -1.0f;
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return a / b;
}

inline float fast_sigmoid_contfrac7(float x) {
  if (x >= 10.0f) return 1.0f;
  if (x <= -10.0f) return 0.0f;
  x = 0.5f * x;
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return 0.5f * (a / b + 1);
}

inline float Sigmoid_Exact(float input) {
  double exp_value;
  double return_value;

  exp_value = exp((double)-input);

  return_value = 1 / (1 + exp_value);

  return (float)return_value;
}

// The hard sigmoid function used by GRU in Keras as a default setting.
inline float Sigmoid_Hard(float input) {
  if (input <= -2.5f)
    return 0.0f;
  else if (input >= 2.5f)
    return 1.0f;
  else
    return 0.2f * input + 0.5f;
}

inline void LSTM_activation_contfrac5_avx(
    const float* pi1, const float* pi2, const float* pbi,
    const float* pf1, const float* pf2, const float* pbf,
    const float* po1, const float* po2, const float* pbo,
    const float* pg1, const float* pg2, const float* pbg,
    float* pcprev, float* pout, int c) {
  float* poutLim = pout + c;
  float* poutLim8 = poutLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (; pout < poutLim8; pcprev += 8, pout += 8) {
    __m256 xi = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(pi1), _mm256_loadu_ps(pi2)), _mm256_loadu_ps(pbi));
    __m256 xigate;
    __Fast_Sigmoid_Contfrac5_Avx__(xi, xigate);

    __m256 xf = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(pf1), _mm256_loadu_ps(pf2)), _mm256_loadu_ps(pbf));
    __m256 xfgate;
    __Fast_Sigmoid_Contfrac5_Avx__(xf, xfgate);

    __m256 xo = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(po1), _mm256_loadu_ps(po2)), _mm256_loadu_ps(pbo));
    __m256 xogate;
    __Fast_Sigmoid_Contfrac5_Avx__(xo, xogate);

    __m256 xg = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(pg1), _mm256_loadu_ps(pg2)), _mm256_loadu_ps(pbg));
    __m256 xggate;
    __Fast_Tanh_Contfrac5_Avx__(xg, xggate);

    __m256 cprev = _mm256_loadu_ps(pcprev);
    __m256 ccurr = _mm256_add_ps(_mm256_mul_ps(cprev, xfgate), _mm256_mul_ps(xigate, xggate));
    _mm256_storeu_ps(pcprev, ccurr);

    __m256 ctanh;
    __Fast_Tanh_Contfrac5_Avx__(ccurr, ctanh);
    __m256 out = _mm256_mul_ps(ctanh, xogate);
    _mm256_storeu_ps(pout, out);

    pi1 += 8;
    pi2 += 8;
    pbi += 8;
    pf1 += 8;
    pf2 += 8;
    pbf += 8;
    po1 += 8;
    po2 += 8;
    pbo += 8;
    pg1 += 8;
    pg2 += 8;
    pbg += 8;
  }

  for (; pout < poutLim; pcprev++, pout++) {
    *pcprev = (*pcprev) * SIGMOID5(*(pf1++) + *(pf2++) + *(pbf++)) + TANH5(*(pg1++) + *(pg2++) + *(pbg++)) * SIGMOID5(*(pi1++) + *(pi2++) + *(pbi++));
    *pout = TANH5(*pcprev) * SIGMOID5(*(po1++) + *(po2++) + *(pbo++));
  }
}
inline void LSTM_activation_contfrac5_avx_v2(const float* pin, const float* pbias_i, const float* pbias_f, const float* pbias_o, const float* pbias_g,
                                             float* pout, float* pmem, int nrow, int ncol) {
  // Double check layout.
  int ncol3 = 3 * ncol;
  int iLim8 = ncol - 7;
  float* pi = (float*)pin;
  float* pf = pi + ncol;
  float* po = pf + ncol;
  float* pg = po + ncol;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (int iRow = 0; iRow < nrow; iRow++, pi += ncol3, pf += ncol3, po += ncol3, pg += ncol3) {
    float* pbi = (float*)pbias_i;
    float* pbf = (float*)pbias_f;
    float* pbo = (float*)pbias_o;
    float* pbg = (float*)pbias_g;

    int i = 0;
    for (; i < iLim8; i += 8, pi += 8, pf += 8, po += 8, pg += 8, pbi += 8, pbf += 8, pbo += 8, pbg += 8, pout += 8, pmem += 8) {
      // Computing sigmoid for input.
      __m256 xi = _mm256_loadu_ps(pi);
      __m256 bi = _mm256_loadu_ps(pbi);
      xi = _mm256_add_ps(xi, bi);
      __m256 xigate;
      __Fast_Sigmoid_Contfrac5_Avx__(xi, xigate);

      // computing sigmoid for forget.
      __m256 xf = _mm256_loadu_ps(pf);
      __m256 bf = _mm256_loadu_ps(pbf);
      xf = _mm256_add_ps(xf, bf);
      __m256 xfgate;
      __Fast_Sigmoid_Contfrac5_Avx__(xf, xfgate);

      // computing sigmoid for output.
      __m256 xo = _mm256_loadu_ps(po);
      __m256 bo = _mm256_loadu_ps(pbo);
      xo = _mm256_add_ps(xo, bo);
      __m256 xogate;
      __Fast_Sigmoid_Contfrac5_Avx__(xo, xogate);

      // computing tanh of gate.
      __m256 xg = _mm256_loadu_ps(pg);
      __m256 bg = _mm256_loadu_ps(pbg);
      xg = _mm256_add_ps(xg, bg);
      __m256 xggate;
      __Fast_Tanh_Contfrac5_Avx__(xg, xggate);

      // computing cell memory
      __m256 cprev = _mm256_loadu_ps(pmem);
      __m256 ccurr = _mm256_add_ps(_mm256_mul_ps(cprev, xfgate), _mm256_mul_ps(xggate, xigate));

      // store cell memory
      _mm256_storeu_ps(pmem, ccurr);

      // computing tanh of cell memory
      __m256 ctanh;
      __Fast_Tanh_Contfrac5_Avx__(ccurr, ctanh); /* Warning: ccurr changes value after this point */
      __m256 out = _mm256_mul_ps(ctanh, xogate);

      // store output
      _mm256_storeu_ps(pout, out);
    }

    for (; i < ncol; i++, pi++, pf++, po++, pg++, pbi++, pbf++, pbo++, pbg++, pout++, pmem++) {
      *pmem = (*pmem) * SIGMOID5((*pf) + *(pbf)) + TANH5((*pg) + (*pbg)) * SIGMOID5((*pi) + (*pbi));
      *pout = TANH5(*pmem) * SIGMOID5((*po) + (*pbo));
    }
  }
}

inline void LSTM_activation_contfrac7_avx_v2(const float* pin, const float* pbias_i, const float* pbias_f, const float* pbias_o, const float* pbias_g,
                                             float* pout, float* pmem, int nrow, int ncol) {
  // Double check layout.
  int ncol3 = 3 * ncol;
  int iLim8 = ncol - 7;
  float* pi = (float*)pin;
  float* pf = pi + ncol;
  float* po = pf + ncol;
  float* pg = po + ncol;

  __Init_Contfrac7_Avx_Constants__;
  __Init_Contfrac7_Avx_Bounds__;

  for (int iRow = 0; iRow < nrow; iRow++, pi += ncol3, pf += ncol3, po += ncol3, pg += ncol3) {
    float* pbi = (float*)pbias_i;
    float* pbf = (float*)pbias_f;
    float* pbo = (float*)pbias_o;
    float* pbg = (float*)pbias_g;

    int i = 0;
    for (; i < iLim8; i += 8, pi += 8, pf += 8, po += 8, pg += 8, pbi += 8, pbf += 8, pbo += 8, pbg += 8, pout += 8, pmem += 8) {
      // Computing sigmoid for input.
      __m256 xi = _mm256_loadu_ps(pi);
      __m256 bi = _mm256_loadu_ps(pbi);
      xi = _mm256_add_ps(xi, bi);
      __m256 xigate;
      __Fast_Sigmoid_Contfrac7_Avx__(xi, xigate);

      // computing sigmoid for forget.
      __m256 xf = _mm256_loadu_ps(pf);
      __m256 bf = _mm256_loadu_ps(pbf);
      xf = _mm256_add_ps(xf, bf);
      __m256 xfgate;
      __Fast_Sigmoid_Contfrac7_Avx__(xf, xfgate);

      // computing sigmoid for output.
      __m256 xo = _mm256_loadu_ps(po);
      __m256 bo = _mm256_loadu_ps(pbo);
      xo = _mm256_add_ps(xo, bo);
      __m256 xogate;
      __Fast_Sigmoid_Contfrac7_Avx__(xo, xogate);

      // computing tanh of gate.
      __m256 xg = _mm256_loadu_ps(pg);
      __m256 bg = _mm256_loadu_ps(pbg);
      xg = _mm256_add_ps(xg, bg);
      __m256 xggate;
      __Fast_Tanh_Contfrac7_Avx__(xg, xggate);

      // computing cell memory
      __m256 cprev = _mm256_loadu_ps(pmem);
      __m256 ccurr = _mm256_add_ps(_mm256_mul_ps(cprev, xfgate), _mm256_mul_ps(xggate, xigate));

      // store cell memory
      _mm256_storeu_ps(pmem, ccurr);

      // computing tanh of cell memory
      __m256 ctanh;
      __Fast_Tanh_Contfrac7_Avx__(ccurr, ctanh); /* Warning: ccurr changes value after this point */
      __m256 out = _mm256_mul_ps(ctanh, xogate);

      // store output
      _mm256_storeu_ps(pout, out);
    }

    for (; i < ncol; i++, pi++, pf++, po++, pg++, pbi++, pbf++, pbo++, pbg++, pout++, pmem++) {
      *pmem = (*pmem) * SIGMOID7((*pf) + *(pbf)) + TANH7((*pg) + (*pbg)) * SIGMOID7((*pi) + (*pbi));
      *pout = TANH7(*pmem) * SIGMOID7((*po) + (*pbo));
    }
  }
}

inline void output_gate_contfrac5_avx_v2(const float* ps11, const float* ps12, const float* ps21, const float* ps22, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (; pd < pdLim8; ps11 += 8, ps12 += 8, ps21 += 8, ps22 += 8, ps4 += 8, pd += 8) {
    __m256 x11 = _mm256_loadu_ps(ps11);
    __m256 x12 = _mm256_loadu_ps(ps12);
    __m256 x21 = _mm256_loadu_ps(ps21);
    __m256 x22 = _mm256_loadu_ps(ps22);
    __m256 s_prev = _mm256_loadu_ps(ps4);

    x11 = _mm256_add_ps(x11, x12);
    x21 = _mm256_add_ps(x21, x22);
    __m256 x13, x23;

    __Fast_Sigmoid_Contfrac5_Avx__(x11, x13);
    __Fast_Tanh_Contfrac5_Avx__(x21, x23);

    __m256 x14 = _mm256_mul_ps(x13, s_prev);
    x13 = _mm256_sub_ps(c1, x13);
    x23 = _mm256_mul_ps(x13, x23);
    x23 = _mm256_add_ps(x23, x14);

    _mm256_storeu_ps(pd, x23);
  }

  for (; pd < pdLim;) {
    float tmp_z = fast_sigmoid_contfrac5((*(ps11++)) + (*(ps12++)));
    float tmp_h = fast_tanh_contfrac5((*(ps21++)) + (*(ps22++)));
    *(pd++) = (1 - tmp_z) * tmp_h + tmp_z * (*(ps4++));
  }
}

inline void reset_gate_contfrac5_avx_v2(const float* ps1, const float* ps2, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;

  for (; pd < pdLim8; ps1 += 8, ps2 += 8, ps4 += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps1);
    __m256 x2 = _mm256_loadu_ps(ps2);
    __m256 s_prev = _mm256_loadu_ps(ps4);
    x1 = _mm256_add_ps(x1, x2);
    __m256 x3;
    __Fast_Sigmoid_Contfrac5_Avx__(x1, x3);
    x3 = _mm256_mul_ps(x3, s_prev);
    _mm256_storeu_ps(pd, x3);
  }

  for (; pd < pdLim;) {
    *(pd++) = (*(ps4++)) * fast_sigmoid_contfrac5((*(ps1++)) + (*(ps2++)));
  }
}

inline void output_gate_hard5_avx_v2(const float* ps11, const float* ps12, const float* ps21, const float* ps22, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Contfrac5_Avx_Bounds__;
  __Init_Hard_Sigmoid_Avx_Bounds__;

  for (; pd < pdLim8; ps11 += 8, ps12 += 8, ps21 += 8, ps22 += 8, ps4 += 8, pd += 8) {
    __m256 x11 = _mm256_loadu_ps(ps11);
    __m256 x12 = _mm256_loadu_ps(ps12);
    __m256 x21 = _mm256_loadu_ps(ps21);
    __m256 x22 = _mm256_loadu_ps(ps22);
    __m256 s_prev = _mm256_loadu_ps(ps4);

    x11 = _mm256_add_ps(x11, x12);
    __m256 x13;
    __Hard_Sigmoid_Avx__(x11, x13);

    x21 = _mm256_add_ps(x21, x22);
    __m256 x23;
    __Fast_Tanh_Contfrac5_Avx__(x21, x23);

    __m256 x14 = _mm256_mul_ps(x13, s_prev);
    x13 = _mm256_sub_ps(c1, x13);
    x23 = _mm256_mul_ps(x13, x23);
    x23 = _mm256_add_ps(x23, x14);

    _mm256_storeu_ps(pd, x23);
  }

  for (; pd < pdLim;) {
    float tmp_z = Sigmoid_Hard((*(ps11++)) + (*(ps12++)));
    float tmp_h = fast_tanh_contfrac5((*(ps21++)) + (*(ps22++)));
    *(pd++) = (1 - tmp_z) * tmp_h + tmp_z * (*(ps4++));
  }
}

inline void reset_gate_hard5_avx_v2(const float* ps1, const float* ps2, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_Contfrac_Avx_Constants__;
  __Init_Hard_Sigmoid_Avx_Bounds__;

  for (; pd < pdLim8; ps1 += 8, ps2 += 8, ps4 += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps1);
    __m256 x2 = _mm256_loadu_ps(ps2);
    __m256 s_prev = _mm256_loadu_ps(ps4);
    x1 = _mm256_add_ps(x1, x2);
    __Hard_Sigmoid_Avx__(x1, x1);
    x1 = _mm256_mul_ps(x1, s_prev);
    _mm256_storeu_ps(pd, x1);
  }

  for (; pd < pdLim;) {
    *(pd++) = (*(ps4++)) * Sigmoid_Hard((*(ps1++)) + (*(ps2++)));
  }
}

inline void reset_gate_contfrac7_avx_v2(const float* ps1, const float* ps2, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __m256 chalf = _mm256_set1_ps(0.5f);
  __m256 c1 = _mm256_set1_ps(1.0f);
  __m256 c28 = _mm256_set1_ps(28.0f);
  __m256 c378 = _mm256_set1_ps(378.0f);
  __m256 c3150 = _mm256_set1_ps(3150.0f);
  __m256 c17325 = _mm256_set1_ps(17325.0f);
  __m256 c62370 = _mm256_set1_ps(62370.0f);
  __m256 c135135 = _mm256_set1_ps(135135.0f);
  __m256 ub = _mm256_set1_ps(5.0f);
  __m256 lb = _mm256_set1_ps(-5.0f);

  for (; pd < pdLim8; ps1 += 8, ps2 += 8, ps4 += 8, pd += 8) {
    __m256 x1 = _mm256_loadu_ps(ps1);
    __m256 x2 = _mm256_loadu_ps(ps2);
    __m256 s_prev = _mm256_loadu_ps(ps4);
    x1 = _mm256_add_ps(x1, x2);
    x1 = _mm256_mul_ps(x1, chalf);
    x1 = _mm256_min_ps(x1, ub);
    x1 = _mm256_max_ps(x1, lb);
    x2 = _mm256_mul_ps(x1, x1);

    //float a = x * (945.0f + x2 * (105.0f + x2));
    //float b = 945.0f + x2 * (420.0f + x2 * 15.0f);
    __m256 x4 = _mm256_add_ps(x2, c378);
    x4 = _mm256_mul_ps(x4, x2);
    x4 = _mm256_add_ps(x4, c17325);
    x4 = _mm256_mul_ps(x2, x4);
    x4 = _mm256_add_ps(x4, c135135);
    x4 = _mm256_mul_ps(x4, x1);

    __m256 x3 = _mm256_mul_ps(x2, c28);
    x3 = _mm256_add_ps(x3, c3150);
    x3 = _mm256_mul_ps(x3, x2);
    x3 = _mm256_add_ps(x3, c62370);
    x3 = _mm256_mul_ps(x3, x2);
    x3 = _mm256_add_ps(x3, c135135);

    x3 = _mm256_div_ps(x4, x3);
    x3 = _mm256_add_ps(x3, c1);
    x3 = _mm256_mul_ps(x3, chalf);

    x3 = _mm256_mul_ps(x3, s_prev);

    _mm256_storeu_ps(pd, x3);
  }

  for (; pd < pdLim;) {
    *(pd++) = (*(ps4++)) * fast_sigmoid_contfrac7((*(ps1++)) + (*(ps2++)));
  }
}

inline void output_gate_contfrac7_avx_v2(const float* ps11, const float* ps12, const float* ps21, const float* ps22, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;
  __m256 chalf = _mm256_set1_ps(0.5f);
  __m256 c1 = _mm256_set1_ps(1.0f);
  __m256 c28 = _mm256_set1_ps(28.0f);
  __m256 c378 = _mm256_set1_ps(378.0f);
  __m256 c3150 = _mm256_set1_ps(3150.0f);
  __m256 c17325 = _mm256_set1_ps(17325.0f);
  __m256 c62370 = _mm256_set1_ps(62370.0f);
  __m256 c135135 = _mm256_set1_ps(135135.0f);
  __m256 ub = _mm256_set1_ps(5.0f);
  __m256 lb = _mm256_set1_ps(-5.0f);

  for (; pd < pdLim8; ps11 += 8, ps12 += 8, ps21 += 8, ps22 += 8, ps4 += 8, pd += 8) {
    __m256 x11 = _mm256_loadu_ps(ps11);
    __m256 x12 = _mm256_loadu_ps(ps12);
    __m256 x21 = _mm256_loadu_ps(ps21);
    __m256 x22 = _mm256_loadu_ps(ps22);
    __m256 s_prev = _mm256_loadu_ps(ps4);

    x11 = _mm256_add_ps(x11, x12);
    x11 = _mm256_mul_ps(x11, chalf);

    x21 = _mm256_add_ps(x21, x22);

    x11 = _mm256_min_ps(x11, ub);
    x11 = _mm256_max_ps(x11, lb);
    x21 = _mm256_min_ps(x21, ub);
    x21 = _mm256_max_ps(x21, lb);

    x12 = _mm256_mul_ps(x11, x11);
    x22 = _mm256_mul_ps(x21, x21);

    //float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    //float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    __m256 x14 = _mm256_add_ps(x12, c378);
    x14 = _mm256_mul_ps(x14, x12);
    x14 = _mm256_add_ps(x14, c17325);
    x14 = _mm256_mul_ps(x12, x14);
    x14 = _mm256_add_ps(x14, c135135);
    x14 = _mm256_mul_ps(x14, x11);

    __m256 x13 = _mm256_mul_ps(x12, c28);
    x13 = _mm256_add_ps(x13, c3150);
    x13 = _mm256_mul_ps(x13, x12);
    x13 = _mm256_add_ps(x13, c62370);
    x13 = _mm256_mul_ps(x13, x12);
    x13 = _mm256_add_ps(x13, c135135);

    x13 = _mm256_div_ps(x14, x13);
    x13 = _mm256_add_ps(x13, c1);
    x13 = _mm256_mul_ps(x13, chalf);

    __m256 x24 = _mm256_add_ps(x22, c378);
    x24 = _mm256_mul_ps(x24, x22);
    x24 = _mm256_add_ps(x24, c17325);
    x24 = _mm256_mul_ps(x24, x22);
    x24 = _mm256_add_ps(x24, c135135);
    x24 = _mm256_mul_ps(x21, x24);

    __m256 x23 = _mm256_mul_ps(x22, c28);
    x23 = _mm256_add_ps(x23, c3150);
    x23 = _mm256_mul_ps(x23, x22);
    x23 = _mm256_add_ps(x23, c62370);
    x23 = _mm256_mul_ps(x23, x22);
    x23 = _mm256_add_ps(x23, c135135);

    x23 = _mm256_div_ps(x24, x23);

    x14 = _mm256_mul_ps(x13, s_prev);
    x13 = _mm256_sub_ps(c1, x13);
    x23 = _mm256_mul_ps(x13, x23);
    x23 = _mm256_add_ps(x23, x14);

    _mm256_storeu_ps(pd, x23);
  }

  for (; pd < pdLim;) {
    float tmp_z = fast_sigmoid_contfrac7((*(ps11++)) + (*(ps12++)));
    float tmp_h = fast_tanh_contfrac7((*(ps21++)) + (*(ps22++)));
    *(pd++) = (1 - tmp_z) * tmp_h + tmp_z * (*(ps4++));
  }
}

inline void reset_gate_exact_v2(const float* ps1, const float* ps2, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  for (; pd < pdLim;) {
    *(pd++) = (*(ps4++)) * Sigmoid_Exact((*(ps1++)) + (*(ps2++)));
  }
}

inline void output_gate_exact_v2(const float* ps11, const float* ps12, const float* ps21, const float* ps22, const float* ps4, float* pd, int c) {
  float* pdLim = pd + c;
  for (; pd < pdLim;) {
    float tmp_z = Sigmoid_Exact((*(ps11++)) + (*(ps12++)));
    float tmp_h = tanh((*(ps21++)) + (*(ps22++)));
    *(pd++) = (1 - tmp_z) * tmp_h + tmp_z * (*(ps4++));
  }
}

inline void relu_avx(const float* ps1, const float* ps2, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_ReLu_Avx__;

  for (; pd < pdLim8; ps1 += 8, ps2 += 8, pd += 8) {
    __m256 x = _mm256_add_ps(_mm256_loadu_ps(ps1), _mm256_loadu_ps(ps2));
    __m256 y;
    __ReLu__Avx(x, y);
    _mm256_storeu_ps(pd, y);
  }

  for (; pd < pdLim; ps1++, ps2++, pd++) {
    float tmp = (*ps1) + (*ps2);
    *pd = tmp > 0 ? tmp : 0.0f;
  }
}

inline void relu_avx(const float* ps, float* pd, int c) {
  float* pdLim = pd + c;
  float* pdLim8 = pdLim - 7;

  __Init_ReLu_Avx__;

  for (; pd < pdLim8; ps += 8, pd += 8) {
    __m256 x = _mm256_loadu_ps(ps);
    __m256 y;
    __ReLu__Avx(x, y);
    _mm256_storeu_ps(pd, y);
  }

  for (; pd < pdLim; ps++, pd++) {
    float tmp = (*ps);
    *pd = tmp > 0 ? tmp : 0.0f;
  }
}
