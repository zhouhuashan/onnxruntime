//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: low_level_func.cpp
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include "low_level_func.h"
#include <fstream>
#include <assert.h>
#include <string>
#include <vector>

namespace
{
	template<typename T>
	void LoadTFDataImp(std::string const &filename, T* dest, int dimensionCount, ...) {
		va_list paramList;
		va_start(paramList, dimensionCount);
		VLoadTFDataImp(filename, dest, dimensionCount, paramList);
		va_end(paramList);
	}

	template<typename T>
	void VLoadTFDataImp(std::string const &filename, T* dest, int dimensionCount, va_list paramList) {
		std::ifstream in_file;
		in_file.open(filename);

		if (in_file.fail()) {
			std::string errMsg = "failed to open " + filename;
			std::cout << errMsg << std::endl;
			throw errMsg;
		}
		else {

			int ldim, ldN;
			int totalNum = 1;
			assert(dimensionCount > 0);
			in_file >> ldim;
			assert(ldim == dimensionCount);
			for (int i = 0; i < dimensionCount; i++)
			{
				in_file >> ldN;
				int dN = va_arg(paramList, int);
				assert(ldN == dN);
				totalNum *= ldN;
			}
			for (int i = 0; i < totalNum; i++)
				in_file >> dest[i];

			in_file.close();
		}
	}

	template<typename T>
	void VLoadTFBinaryImp(std::string const &filename, T* dest, int dimensionCount, va_list paramList) {

		std::ifstream in_file(filename, std::fstream::binary | std::fstream::in);

		if (in_file.fail()) {
			std::string errMsg = "failed to open " + filename;
			std::cout << errMsg << std::endl;
			throw errMsg;
		}
		else {
            std::vector<int> shape;
            int ldim;
			assert(dimensionCount > 0);
            in_file.read((char*)&ldim, sizeof(ldim));
			assert(ldim == dimensionCount);

            shape.resize(ldim);
            in_file.read((char*)shape.data(), sizeof(int)*ldim);

			int totalNum = 1;
			for (int i = 0; i < dimensionCount; i++)
			{
				int dN = va_arg(paramList, int);
				assert(shape[i] == dN);
				totalNum *= dN;
			}

			in_file.read((char*)dest, sizeof(T)*totalNum);
			in_file.close();
		}
	}
}

void AssignRandomNumber(float *x, int length) {
	AssignRandomNumber(x, length, -0.5f, 0.5f);
}


void ArraySlicing(float *x, int row, int col, int selected_col, float* output) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (j == selected_col) {
				output[i] = x[i * col + j];
			}			
		}
	}
}

void WhereGT(float *lhs, float* rhs, float* candidate1, float* candidate2, int len, float* output) {
	for (int i = 0; i < len; i++) {
		if (lhs[i] > rhs[i]) {
			output[i] = candidate1[i];
		}
		else {
			output[i] = candidate2[i];
		}
	}
}

void Maximum(float *lhs, float* rhs, int len, float* output) {
	for (int i = 0; i < len; i++) {
		if (lhs[i] >= rhs[i]) {
			output[i] = lhs[i];
		}
		else {
			output[i] = rhs[i];
		}
	}
}

void ReduceSum(float *input, int Ni, int Nj, int Nk, int axis, float* output) {
	
	float sum;
	if (axis == 1) {
		for (int i = 0; i < Ni; i++) {
			for (int k = 0; k < Nk; k++) {
				sum = 0;
				for (int j = 0; j < Nj; j++) {
					//sum += input[i][j][k];
					sum += input[i * Nj * Nk + j * Nk + k];
				}
				//output[i][k] = sum;
				output[i * Nk + k] = sum;
			}
		}
	}
	else {
		printf("Reduce sum along axis %d: Not Supported", axis);
		assert(false);
	}
}

// Implement sigmoid function: S(t) = 1 / (1 + e^-t)
// The underlying code will return an exact 0 or 1 if an element of x is too small or too big.
float Sigmoid(float x) {

    double exp_value;
    float return_value;

    exp_value = exp((double)-x);

    return_value = 1 / (1 + exp_value);
    return return_value;
}

//Read here for a simplified version of softmax 
//http://stackoverflow.com/questions/34968722/softmax-function-python
// A numerically stable variant of the softmax
// Note: values of x are changed. 
void Softmax(float *x, float *y, int len) {

    assert(len >= 1);
    double max = x[0];
    double sum = 0.0;

    for (int i = 1; i < len; i++) {
        if (max < x[i]) {
            max = x[i];
        }
    }

    for (int i = 0; i < len; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < len; i++) {
        y[i] = x[i] / sum;
    }
}

void BasicSoftmax(const float *x, float *y, int len) {

	assert(len >= 1);
	double sum = 0.0;

	for (int i = 0; i < len; i++) {
		y[i] = exp(x[i]);
		sum += y[i];
	}

	if (sum == 0) {
		for (int i = 0; i < len; i++) {
			y[i] = 1.0 / (float)len;
		}
	}
	else {
		for (int i = 0; i < len; i++) {
			y[i] = y[i] / sum;
		}
	}
}

//prints 2 matrix when the number of cols is not equal to the col_width of the src
void printMatrix(std::string name, float* src, int row, int col, int offset, int col_width) {

    std::cout << " Printing Matrix :" << name << std::endl << std::endl;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            int index = r * col_width + offset + c;
            std::cout << std::setw(10) << std::setprecision(4) << src[index];
        }
        std::cout << std::endl << std::endl;
    }

}

//prints 2 matrix when the number of cols is not equal to the col_width of the src
void printMatrix(std::string name, int* src, int row, int col, int offset, int col_width) {

	std::cout << " Printing Matrix :" << name << std::endl << std::endl;
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			int index = r * col_width + offset + c;
			std::cout << std::setw(10) << std::setprecision(4) << src[index];
		}
		std::cout << std::endl << std::endl;
	}

}


void printAddedMatrix(std::string name, int* src1, float* src2, int row, int col, int offset, int col_width) {

    std::cout << " Printing Matrix :" << name << std::endl << std::endl;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            int index = r * col_width + offset + c;
            std::cout << std::setw(10) << std::setprecision(4) << src1[index] + src2[index];
        }
        std::cout << std::endl << std::endl;
    }

}

void PrintMatrix1D(float *x, int length) {
    printf("[");
    for (int i = 0; i < length; i++) {
        printf("%f, ", x[i]);
    }
    printf("]\n");
}

void PrintMatrix2D(float *x, int x_length, int y_length) {
    printf("[");
    for (int i = 0; i < x_length; i++) {
		std::cout << " row :" << i<< " ";
		printf("{");
        for (int j = 0; j < y_length; j++)
        {
            printf("%f, ", x[i * y_length + j]);
        }
        printf("},");
        printf("\n");
    }
    printf("]\n");
}

void PrintMatrix2DCorners(float *x, int x_length, int y_length) {
    printf("[%.6f,    %.6f,    %.6f,    %.6f]\r\n", *x, *(x + y_length - 1), *(x + (x_length - 1) * y_length), *(x + (x_length * y_length - 1)));
}

void PrintMatrix3D(float *x, int x_length, int y_length, int z_length) {

    if (x_length <= 10 && y_length <= 10 && z_length <= 10) {
        printf("[");
        for (int i = 0; i < x_length; i++) {
            if (i == 0 || i == x_length - 1) {
                printf("{");
                for (int j = 0; j < y_length; j++)
                {
                    printf("{");
                    for (int k = 0; k < z_length; k++) {
                        printf("%.10f, ", x[i * y_length * z_length + j * z_length + k]);
                    }
                    printf("},\n");
                }
                printf("},");
                printf("\n");
            }
        }
        printf("]\n");
    }
    else {
        bool isXOmitted = false;
        printf("[");
        for (int i = 0; i < x_length; i++) {
            if (i < 3 || x_length - i <= 3) {
                bool isYOmitted = false;
                printf("{");
                for (int j = 0; j < y_length; j++) {
                    if (j < 3 || y_length - j <= 3) {
                        bool isZOmitted = false;
                        printf("{");
                        for (int k = 0; k < z_length; k++) {
                            if (k < 3 || z_length - k <= 3) {
                                printf("%.10f, ", x[i * y_length * z_length + j * z_length + k]);
                            }
                            else {
                                if (!isZOmitted) {
                                    printf("...,");
                                    isZOmitted = true;
                                }
                            }
                        }
                        printf("},\n");
                    }
                    else {
                        if (!isYOmitted) {
                            printf("...,\n");
                            isYOmitted = true;
                        }
                    }
                }
                printf("},");
                printf("\n");
            }
            else {
                if (!isXOmitted) {
                    printf("...\n");
                    isXOmitted = true;
                }
            }
        }
        printf("]\n");
    }
}

float * CreateMatrices(int64_t num_arrays, int64_t size) {
    float* new_array;
#ifdef USE_MKL
    new_array = (float *)mkl_malloc((int64_t)num_arrays * size * sizeof(float), 64);
#else
    new_array = new float[num_arrays * size];
#endif
    return new_array;
}

int32_t * CreateInt32Matrices(int64_t num_arrays, int64_t size) {
    int32_t* new_array;
#ifdef USE_MKL
    new_array = (int32_t *)mkl_malloc((int64_t)num_arrays * size * sizeof(int32_t), 64);
#else
    new_array = new int32_t[num_arrays * size];
#endif
    return new_array;
}

void RepeatVectorToConstructArray(float* vector, int len_vector, float* arr, int reps) {
    for (int i = 0; i < reps; i++) {
        ArrayCopy(vector, len_vector, arr + i * len_vector);
    }
}

int forceAffinity(int i)
{
    kmp_affinity_mask_t mask;

    kmp_create_affinity_mask(&mask);
    kmp_set_affinity_mask_proc(i, &mask);

    return (kmp_set_affinity(&mask) == 0);
}

void SaveWeightsToFile(std::ofstream& output_file, float * weight_matrix, int size) {
    for (int count = 0; count < size; count++)
    {
        output_file << weight_matrix[count] << " ";
    }
    output_file << std::endl;
}

void SaveToFile(std::string const &filename, float * matrix, int size) {
    std::ofstream out_file;
    out_file.open(filename);

    if (out_file.fail()) {
        std::cout << "File opening error" << std::endl;
    }
    else {

        for (int count = 0; count < size; count++)
        {
            out_file << matrix[count] << " ";
        }
        out_file << std::endl;
        out_file << std::endl;

        out_file.close();
    }

}

void SaveWeightsCornersToFile(std::ofstream& output_file, float * weight_matrix, int nrow, int ncol) {
    output_file << "[" << *weight_matrix << "    " << *(weight_matrix + ncol - 1) << "    " << *(weight_matrix + (nrow - 1)*ncol) << "    " << *(weight_matrix + (nrow * ncol - 1)) << "]\r\n";
}

void RestoreWeightsToFile(std::ifstream& input_file, float * weight_matrix, int size) {
    for (int count = 0; count < size; count++)
    {
        input_file >> weight_matrix[count];
    }
}

void CopyMatrix(const float * src, float * dst, int nrow, int ncol) {
    for (int i = 0; i < nrow * ncol; i++) {
        *(dst++) = *(src++);
    }
}

void CopyMatrixBackToBack(const float * src, float * dst1, float * dst2, int nrow, int ncol) {
    int d_offset = 0;
    int s1_offset = 0;
    int s2_offset = ncol;
    for (int iRow = 0; iRow < nrow; iRow++) {
        for (int iCol = 0; iCol < ncol; iCol++) {
            dst1[d_offset] = src[s1_offset++];
            dst2[d_offset] = src[s2_offset++];
            d_offset++;
        }

        s1_offset += ncol;
        s2_offset += ncol;
    }
}

void CopyMatrixWithTranspose(const float * src, float * dst, int nrow, int ncol) {
    int s_offset = 0;
    int d_offset = 0;
    for (int iRow = 0; iRow < nrow; iRow++) {
        s_offset = iRow;
        for (int iCol = 0; iCol < ncol; iCol++) {
            dst[d_offset++] = src[s_offset];
            s_offset += nrow;
        }
    }
    assert(s_offset == nrow * ncol + nrow - 1);
    assert(d_offset == nrow * ncol);
}

void CopyMatrixWithTransposeSkip(const float * src, float * dst, int nSeq, int nBatch, int dim) {
    int s_giant_step = 0;
    int s_offset = 0;
    int d_offset = 0;
    for (int iSeq = 0; iSeq < nSeq; iSeq++) {
        s_giant_step = iSeq * dim;
        for (int iBatch = 0; iBatch < nBatch; iBatch++) {
            s_offset = s_giant_step;
            for (int i = 0; i < dim; i++) {
                dst[d_offset++] = src[s_offset++];
            }
            // Go to next batch.
            s_giant_step += nSeq * dim;
        }
    }
}

//Sequence, Batch, Dim to Batch, Sequence, Dim
void TransposeSBDtoBSD(const float* src, float* dest, int nSeq, int nBatch, int dim) {
	int index = 0;
	for (int s = 0; s < nSeq; s++) {
		for (int b = 0; b < nBatch; b++) {
			for (int d = 0; d < dim; d++) {
				dest[b * nSeq * dim + s * dim + d] = src[index++];
			}
		}
	}

}

//Batch, Sequence, Dim to Sequence, Batch, Dim
void TransposeBSDtoSBD(const float* src, float* dest, int nBatch, int nSeq, int dim) {
	int index = 0;
	for (int s = 0; s < nSeq; s++) {
		for (int b = 0; b < nBatch; b++) {
			for (int d = 0; d < dim; d++) {
				dest[index++] = src[b * nSeq * dim + s * dim + d];
			}
		}
	}

}

//Tile Sequence, 1, Dim to Sequence, Batch, Dim
void TileBatch(const float* src, float* dest, int nSeq, int nBatch, int dim) {
	int index = 0;
	for (int s = 0; s < nSeq; s++) {
		for (int b = 0; b < nBatch; b++) {
			for (int d = 0; d < dim; d++) {
				dest[index++] = src[s * dim + d];
			}
		}
	}


}

void Concat2(float* input_a, int row_a, int col_a, float* input_b, int row_b, int col_b, float* output_c, int concat_dim) {
	//  t1 = [[1, 2, 3], [4, 5, 6]]
	//	t2 = [[7, 8, 9], [10, 11, 12]]
	//	tf.concat(0, [t1, t2]) == >[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
	//	tf.concat(1, [t1, t2]) == >[[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

	//  # tensor t3 with shape[2, 3]
	//	# tensor t4 with shape[2, 3]
	//	tf.shape(tf.concat(0, [t3, t4])) == >[4, 3]
	//	tf.shape(tf.concat(1, [t3, t4])) == >[2, 6]

	if (concat_dim == 1) {
		assert(row_a == row_b);
	} else {
		printf("Concatenation along axis %d : Not implemented.", concat_dim);
		assert(false);
	}

	float* src_a;
	float* src_b;
	float* dest;
	for (int i = 0; i < row_a; i++) {
		src_a = input_a + i * col_a;
		src_b = input_b + i * col_b;
		dest = output_c + i * (col_a + col_b);
		std::memcpy(dest, src_a, sizeof(float) * col_a);
		std::memcpy(dest + col_a, src_b, sizeof(float) * col_b);
	}
}

void Concat3(float* input_a, int row_a, int col_a, float* input_b, int row_b, int col_b, float* input_c, int row_c, int col_c, float* output, int concat_dim) {
	//  t1 = [[1, 2, 3], [4, 5, 6]]
	//	t2 = [[7, 8, 9], [10, 11, 12]]
	//	tf.concat(0, [t1, t2]) == >[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
	//	tf.concat(1, [t1, t2]) == >[[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

	//  # tensor t3 with shape[2, 3]
	//	# tensor t4 with shape[2, 3]
	//	tf.shape(tf.concat(0, [t3, t4])) == >[4, 3]
	//	tf.shape(tf.concat(1, [t3, t4])) == >[2, 6]

	if (concat_dim == 1) {
		assert(row_a == row_b);
	}
	else {
		printf("Concatenation along axis %d : Not implemented.", concat_dim);
		assert(false);
	}

	float* src_a;
	float* src_b;
	float* src_c;
	float* dest;
	for (int i = 0; i < row_a; i++) {
		src_a = input_a + i * col_a;
		src_b = input_b + i * col_b;
		src_c = input_c + i * col_c;
		dest = output + i * (col_a + col_b + col_c);
		std::memcpy(dest, src_a, sizeof(float) * col_a);
		std::memcpy(dest + col_a, src_b, sizeof(float) * col_b);
		std::memcpy(dest + col_a + col_b, src_c, sizeof(float) * col_c);
	}
}

void Concat4(float* input_a, int row_a, int col_a, float* input_b, int row_b, int col_b, float* input_c, int row_c, int col_c, float* input_d, int row_d, int col_d, float* output, int concat_dim) {
	//  t1 = [[1, 2, 3], [4, 5, 6]]
	//	t2 = [[7, 8, 9], [10, 11, 12]]
	//	tf.concat(0, [t1, t2]) == >[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
	//	tf.concat(1, [t1, t2]) == >[[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

	//  # tensor t3 with shape[2, 3]
	//	# tensor t4 with shape[2, 3]
	//	tf.shape(tf.concat(0, [t3, t4])) == >[4, 3]
	//	tf.shape(tf.concat(1, [t3, t4])) == >[2, 6]

	if (concat_dim == 1) {
		assert(row_a == row_b);
	}
	else {
		printf("Concatenation along axis %d : Not implemented.", concat_dim);
		assert(false);
	}

	float* src_a;
	float* src_b;
	float* src_c;
	float* src_d;
	float* dest;
	for (int i = 0; i < row_a; i++) {
		src_a = input_a + i * col_a;
		src_b = input_b + i * col_b;
		src_c = input_c + i * col_c;
		src_d = input_d + i * col_d;
		dest = output + i * (col_a + col_b + col_c + col_d);
		std::memcpy(dest, src_a, sizeof(float) * col_a);
		std::memcpy(dest + col_a, src_b, sizeof(float) * col_b);
		std::memcpy(dest + col_a + col_b, src_c, sizeof(float) * col_c);
		std::memcpy(dest + col_a + col_b + col_c, src_d, sizeof(float) * col_d);
	}
}


void Dropout(float* input, int size, float keep_prob) {

	//  With probability `keep_prob`, outputs the input element scaled up by
	//	`1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
	//	sum is unchanged.
	assert(keep_prob > 0);
	assert(keep_prob <= 1);

	if (keep_prob == 1) {
		return;
	}

	float rand_num;
	for (int i = 0; i < size; i++) {
		rand_num = ((float)rand()) / RAND_MAX;
		if (rand_num < keep_prob) {
			input[i] /= keep_prob;
		}
		else {
			input[i] = 0;
		}
	}
}

//comparing an array directly to a intermediate file generated by tensorflow
bool IsSameAsTFData(const float* src, std::string const &filename, float tolerance, int dimension, ...) {
	int* dimensions = new int[dimension];
	int* dimensionMultiple = new int[dimension];
	int length = 1;
	va_list args;
	va_start(args, dimension);
	for (int i = 0; i < dimension; i++)
	{
		dimensions[i] = va_arg(args, int);
		length *= dimensions[i];
	}
	dimensionMultiple[dimension - 1] = 1;
	for (int i = dimension - 2; i >= 0; i--)
	{
		dimensionMultiple[i] = dimensions[i+1] * dimensionMultiple[i+1];
	}
	va_end(args);
	va_start(args, dimension);
	float* tf_data = new float[length];
	VLoadTFDataImp(filename, tf_data, dimension, args);
	va_end(args);

	bool same = true;
	std::cout << "Size : " << length << std::endl;
	for (int i = 0; i < length; i++)
	{
		if (abs(src[i] - tf_data[i]) > tolerance) {
			std::cout << src[i] << " != " << tf_data[i] << " at [";
			int temp = i;
			for (int j = 0; j < dimension ; j++)
			{
				auto current = temp / dimensionMultiple[j];
				temp = temp % dimensionMultiple[j];
				std::cout << current;
				if (j != dimension - 1)
				{
					std::cout << ", ";
				}
			}
			std::cout << "]";
			std::cout << std::endl;
			same = false;
		}
	}
	bool ret_value = same;
	delete[] tf_data;
	delete[] dimensions;
	delete[] dimensionMultiple;
	std::string result = "Passed : ";
	if (!ret_value) result = "Failed :";
	std::cout << result << filename << std::endl;

	return ret_value;
}


void LoadTFData(std::string const &filename, float* dest, int dim, ...) {
	va_list args;
	va_start(args, dim);
	VLoadTFDataImp(filename, dest, dim, args);
	va_end(args);
}

void LoadTFData(std::string const &filename, int* dest, int dim, ...) {
	va_list args;
	va_start(args, dim);
	VLoadTFDataImp(filename, dest, dim, args);
	va_end(args);
}

void LoadTFBinary(std::string const &filename, float* dest, int dim, ...) {
	va_list args;
	va_start(args, dim);
	VLoadTFBinaryImp(filename, dest, dim, args);
	va_end(args);
}

void LoadTFBinary(std::string const &filename, int* dest, int dim, ...) {
	va_list args;
	va_start(args, dim);
	VLoadTFBinaryImp(filename, dest, dim, args);
	va_end(args);
}

bool AreSameArrays(float* src, float* dest, float tolerance, int32_t size) {
	bool same = true;
	std::cout << "Size : "<<size << std::endl;
	for (int i = 0; i < size; i++) {
		if (abs(src[i] - dest[i]) > tolerance) {
			std::cout << i << " : " << src[i] << " != " << dest[i] << std::endl;
			same = false;
		}
	}
	return same;
}