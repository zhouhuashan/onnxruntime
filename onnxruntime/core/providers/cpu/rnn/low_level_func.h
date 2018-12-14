//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: low_level_func.h
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#pragma once
#include <assert.h>
#define USE_MKL
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#define MKL_ALIGN 64
#endif

#include <iomanip>

#ifdef FASTASRSERVINGLIBRARY_EXPORTS  
#define FASTASRSERVINGLIBRARY_API
#else  
#define FASTASRSERVINGLIBRARY_API
#endif

#define __REMOVE


FASTASRSERVINGLIBRARY_API void AssignRandomNumber(float *x, int length);

template<typename T>
void AssignRandomNumber(T* x, int length, T min, T max) {
	assert(max >= min);
	for (int i = 0; i < length; i++) {
		x[i] = (T)((((float)rand()) / (float)RAND_MAX) * ((float)max - (float)min) + (float)min);
	}
}

template<typename T>
void AssignOnes(T *x, int length)
{
		for (int i = 0; i < length; i++) {
			x[i] = 1;
		}
}

template<typename T>
void AssignZeros(T *x, int length)
{
	for (int i = 0; i < length; i++) {
		x[i] = 0;
	}
}

template<typename T>
void ArrayCopy(T* src, int size, T* des)
{
    for (int i = 0; i < size; i++) {
        des[i] = src[i];
    }
}

FASTASRSERVINGLIBRARY_API void ArraySlicing(float *x, int row, int col, int selected_col, float* output);

#ifndef __REMOVE
inline void HadamardProd(float *vector_a, float *vector_b, float *vector_c, int len_batch, int len_dim) {
#ifdef USE_MKL
    vmsMul(len_batch * len_dim, vector_a, vector_b, vector_c, VML_EP);
#else
    int len = len_batch * len_dim;
    for (int i = 0; i < len; i++) {
        vector_c[i] = vector_a[i] * vector_b[i];
    }
#endif
}

inline void ElementwiseAdd(float *vector_a, float *vector_b, float *vector_c, int len_batch, int len_dim) {
#ifdef USE_MKL
    vmsAdd(len_batch * len_dim, vector_a, vector_b, vector_c, VML_EP);
#else
    int len = len_batch * len_dim;
    for (int i = 0; i < len; i++) {
        vector_c[i] = vector_a[i] + vector_b[i];
    }
#endif
}
#endif

// Multiply two matrixes matrix_a[row, inner] X matrix_b[inner, col], and the results are stored into the product array.
inline void MatrixMult(float *matrix_a, float *matrix_b, float *product, int row, int col, int inner) {

#ifdef USE_MKL
    float alpha = 1.0f;
    float beta = 0.0f;
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        row, col, inner, alpha,
        matrix_a, inner,
        matrix_b, col, beta,
        product, col
    );
#else
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            product[i * col + j] = 0;
            for (int k = 0; k < inner; k++) {
                product[i * col + j] += matrix_a[i * inner + k] * matrix_b[k * col + j];
            }
        }
    }
#endif
}

inline void ElementwiseProduct(float *matrix_a, float *matrix_b, float *product, int length) {
	for (int i = 0; i < length; i++) {
		product[i] = matrix_a[i] * matrix_b[i];
	}
}

inline void ScalarVectorProduct(float scalar, float *matrix, float *product, int length) {
	for (int i = 0; i < length; i++) {
		product[i] = matrix[i] * scalar;
	}
}

FASTASRSERVINGLIBRARY_API void WhereGT(float *lhs, float* rhs, float* candidate1, float* candidate2, int len, float* output);

FASTASRSERVINGLIBRARY_API void Maximum(float *lhs, float* rhs, int len, float* output);

FASTASRSERVINGLIBRARY_API void ReduceSum(float *input, int Ni, int Nj, int Nk, int axis, float* output);

float Sigmoid(float x);

FASTASRSERVINGLIBRARY_API void Softmax(float *x, float *y, int len);

FASTASRSERVINGLIBRARY_API void BasicSoftmax(const float *x, float *y, int len);

FASTASRSERVINGLIBRARY_API void printMatrix(std::string name, float* src, int row, int col, int offset, int col_width);

FASTASRSERVINGLIBRARY_API void printMatrix(std::string name, int* src, int row, int col, int offset, int col_width);

FASTASRSERVINGLIBRARY_API void printAddedMatrix(std::string name, float* src1, float* src2, int row, int col, int offset, int col_width);

FASTASRSERVINGLIBRARY_API void PrintMatrix1D(float *x, int length);

FASTASRSERVINGLIBRARY_API void PrintMatrix2D(float *x, int x_length, int y_length);

FASTASRSERVINGLIBRARY_API void PrintMatrix2DCorners(float *x, int x_length, int y_length);

FASTASRSERVINGLIBRARY_API void PrintMatrix3D(float *x, int x_length, int y_length, int z_length);

FASTASRSERVINGLIBRARY_API float * CreateMatrices(int64_t num_arrays, int64_t size);

FASTASRSERVINGLIBRARY_API int32_t * CreateInt32Matrices(int64_t num_arrays, int64_t size);

void RepeatVectorToConstructArray(float* vector, int len_vector, float* arr, int reps);

int forceAffinity(int i);

FASTASRSERVINGLIBRARY_API void SaveToFile(std::string const &filename, float * matrix, int size);

FASTASRSERVINGLIBRARY_API void SaveWeightsToFile(std::ofstream& output_file, float * weight_matrix, int size);

FASTASRSERVINGLIBRARY_API void SaveWeightsCornersToFile(std::ofstream& output_file, float * weight_matrix, int nrow, int ncol);

FASTASRSERVINGLIBRARY_API void RestoreWeightsToFile(std::ifstream& input_file, float * weight_matrix, int size);

void CopyMatrix(const float * src, float * dst, int nrow, int ncol);

void CopyMatrixBackToBack(const float * src, float * dst1, float * dst2, int nrow, int ncol);

void CopyMatrixWithTranspose(const float * src, float * dst, int nrow, int ncol);

//Transpose from Sequence, Batch, Dim to Batch, D Sequence
FASTASRSERVINGLIBRARY_API void TransposeSBDtoBSD(const float* src, float* dest, int nSeq, int nBatch, int dim);
FASTASRSERVINGLIBRARY_API void TransposeBSDtoSBD(const float* src, float* dest, int nBatch, int nSeq, int dim);
FASTASRSERVINGLIBRARY_API void CopyMatrixWithTransposeSkip(const float * src, float * dst, int nSeq, int nBatch, int dim);

//Tile Sequence, 1, Dim to Sequence, Batch, Dim
FASTASRSERVINGLIBRARY_API void TileBatch(const float* src, float* dest, int nSeq, int nBatch, int dim);

//TODO: refactor to use variable parameter function to reduce into 1 funcion
FASTASRSERVINGLIBRARY_API void Concat2(float* input_a, int row_a, int col_a, float* input_b, int row_b, int col_b, float* output_c, int concat_dim);
FASTASRSERVINGLIBRARY_API void Concat3(float* input_a, int row_a, int col_a, float* input_b, int row_b, int col_b, float* input_c, int row_c, int col_c, float* output, int concat_dim);
FASTASRSERVINGLIBRARY_API void Concat4(float* input_a, int row_a, int col_a, float* input_b, int row_b, int col_b, float* input_c, int row_c, int col_c, float* input_d, int row_d, int col_d, float* output, int concat_dim);

FASTASRSERVINGLIBRARY_API void Dropout(float* input, int size, float keep_prob);

FASTASRSERVINGLIBRARY_API void LoadTFData(std::string const &filename, float* dest, int dim, ...);
FASTASRSERVINGLIBRARY_API void LoadTFData(std::string const &filename, int* dest, int dim, ...);

FASTASRSERVINGLIBRARY_API void LoadTFBinary(std::string const &filename, float* dest, int dim, ...);
FASTASRSERVINGLIBRARY_API void LoadTFBinary(std::string const &filename, int* dest, int dim, ...);

// reorder the multi dimension array
// the interface is  inout, dimCount, sizeofdim0, sizeofdim1, sizeofdim2, ... sizeofdimN, newIndexofdim0, newIndexofDim1, ... newIndexOfDimN
// e.g. we want to Transpose a N*M matrix into M*N it would like ReorderMultDimensionArray(matrix, 2, N, M, 1, 0)
// refer to UT for more example
template<typename T>
void ReorderMultDimensionArray(T* inout, int dim, ...)
{
	int* oldDimension = new int[dim];
	int* oldDimensionAggrate = new int[dim];
	int* newDimensionOrder = new int[dim];
	bool* flags = new bool[dim];
	int* newDimensionAggrate = new int[dim];
	int* dimensionIndex = new int[dim];
	int oldTotalCount = 1;
	va_list args;
	va_start(args, dim);
	for (int i = 0; i < dim; i++)
	{
		oldDimension[i] = va_arg(args, int);
		oldTotalCount *= oldDimension[i];
		flags[i] = false;
	}
	
	for (int i = 0; i < dim; i++)
	{
		int newIndex = va_arg(args, int);
		
		assert(newIndex >= 0 && newIndex < dim);
		newDimensionOrder[newIndex] = i;
		flags[i] = true;
	}

	for (int i = 0; i < dim; i++)
	{
		assert(flags[i]);
	}

	va_end(args);
	assert(oldTotalCount > 0);
	newDimensionAggrate[newDimensionOrder[dim-1]] = 1;
	oldDimensionAggrate[dim - 1] = 1;
	for (int i = dim - 2; i >= 0; i--)
	{
		newDimensionAggrate[newDimensionOrder[i]] = newDimensionAggrate[newDimensionOrder[i+1]] * oldDimension[newDimensionOrder[i + 1]];
		oldDimensionAggrate[i] = oldDimensionAggrate[i + 1] * oldDimension[i + 1];
	}
	T* tmp = new T[oldTotalCount];
	memcpy(tmp, inout, sizeof(T) * oldTotalCount);
	for (int i = 0; i < oldTotalCount; i++)
	{
		int index = i;
		for (int j = 0; j < dim; j++)
		{
			dimensionIndex[j] = index / oldDimensionAggrate[j];
			index = index % oldDimensionAggrate[j];
		}
		int newIndex = 0;
		for (int j = 0; j < dim; j++)
		{
			newIndex += dimensionIndex[j] * newDimensionAggrate[j];
		}
		inout[newIndex] = tmp[i];
	}
	delete[] tmp;
	delete[] oldDimension;
	delete[] oldDimensionAggrate;
	delete[] newDimensionOrder;
	delete[] flags;
	delete[] newDimensionAggrate;
	delete[] dimensionIndex;

}


FASTASRSERVINGLIBRARY_API bool AreSameArrays(float* src, float* dest, float tolerance, int32_t size);
FASTASRSERVINGLIBRARY_API bool IsSameAsTFData(const float* src, std::string const &filename, float tolerance, int dimension, ...);

//TODO replace CallocFloatBufferMKL with MemAllocate which returns shared_ptr. 
// And we don't need to worry memory manage issues.
inline float * CallocFloatBufferMKL(size_t num)
{
#ifdef USE_MKL
    return (float *)mkl_calloc(num, sizeof(float), MKL_ALIGN);
#else
    return (float *)calloc(num, sizeof(float));
#endif
}

//TODO, as as above
inline float * CallocFloatBufferMKLWithRandomNumbers(size_t num)
{
    float * ret = CallocFloatBufferMKL(num);
    AssignRandomNumber(ret, (int)num);
    return ret;
}

//TODO, as as above
inline void FreeBufferMKL(float * buf)
{
    if (buf == NULL)
        return;

#ifdef USE_MKL
    mkl_free(buf);
#else
    free(buf);
#endif

    buf = NULL;
}

inline void FreeBuffer(float * buf)
{
    if (buf != NULL)
        free(buf);

    buf = NULL;
}
