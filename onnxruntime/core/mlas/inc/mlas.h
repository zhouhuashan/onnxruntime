/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for the Microsoft Machine Learning algebra subprogram library.

--*/

#pragma once
// clang-format off

#include <stdlib.h>
#include <stdint.h>

//
// Define the calling convention for Windows targets.
//

#if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define MLASCALL __stdcall
#else
#define MLASCALL
#endif

//
// Basic Linear Algebra Subprograms (BLAS) types.
//

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper=121, CblasLower=122 } CBLAS_UPLO;
typedef enum { CblasNonUnit=131, CblasUnit=132 } CBLAS_DIAG;
typedef enum { CblasLeft=141, CblasRight=142} CBLAS_SIDE;
#endif

//
// Single precision matrix/matrix multiply routine.
//

void
MLASCALL
MlasSgemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
    );

//
// Convolution routines.
//

typedef struct _MLAS_CONV_PARAMETERS {
    size_t Dimensions;
    size_t InputChannels;
    size_t InputShape[3];
    size_t KernelShape[3];
    size_t DilationShape[3];
    size_t Padding[6];
    size_t StrideShape[3];
    size_t FilterCount;
    size_t OutputShape[3];
    size_t N;
    size_t K;
    size_t ThreadStrideN;
} MLAS_CONV_PARAMETERS;

bool
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    int64_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    int64_t FilterCount,
    size_t* WorkingBufferSize
    );

#if defined(_M_IX86) || defined(_M_ARM)

inline
bool
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    int64_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    int64_t FilterCount,
    unsigned long* WorkingBufferSize
    )
{
    // Temporary adapter for Lotus which passes WorkingBufferSize as SIZE_T,
    // which in 32-bit Windows expands as "unsigned long" instead of "unsigned
    // int" for size_t. This can be removed when Lotus switches to size_t.
    return MlasConvPrepare(Parameters, Dimensions, InputChannels, InputShape,
        KernelShape, DilationShape, Padding, StrideShape, FilterCount,
        reinterpret_cast<size_t*>(WorkingBufferSize));
}

#endif

void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output
    );

inline
void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* WorkingBuffer,
    float* Output
    )
{
    MlasConv(Parameters, Input, Filter, nullptr, WorkingBuffer, Output);
}

//
// Bias addition routine.
//

void
MLASCALL
MlasBiasAdd(
    const float* Bias,
    size_t M,
    float* Output,
    size_t N,
    size_t ldc
    );

//
// Half-precision floating-point routines.
//

extern "C"
void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const unsigned short* Source,
    float* Destination,
    size_t Count
    );
