//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: gru.cpp
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "low_level_func.h"
#include "activation.h"
#include "gru_fs.h"
#include "lut.h"
#include <fstream>

#define PARALLEL_GRU
#define USE_MKL

#ifdef PARALLEL_GRU
#include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#define USE_AVX 1

#define MAX_NUM_CORES 12

#define RESET_GATE_V2 reset_gate_contfrac5_avx_v2
#define OUTPUT_GATE_V2 output_gate_contfrac5_avx_v2

//#define VERBOSE_GRU_FINAL_OUTPUT 1

GRU::GRU(int input_dim, int input_batch_num, int hidden_dim, int MAX_INPUT_LENGTH, bool is_fw)
    : RNN(input_dim, input_batch_num, hidden_dim, MAX_INPUT_LENGTH, is_fw) {
  /*	_input_dim = input_dim;
        _input_batch_num = input_batch_num;
        _hidden_dim = hidden_dim;
        _MAX_INPUT_LENGTH = MAX_INPUT_LENGTH;
        _is_fw = is_fw;
        */

  /*_fused_hidden_rows = 16;
        _fused_input_rows = 1600;
        _input_matrix_threads = 11;
        _hidden_matrix_threads = 11;
        _threads = 11;

    */
  setFusedRows();
  setRNNType(GRUCell);
  //std::cout << " Fused Input Rows : " << _fused_input_rows << " Fused Hidden Rows : " << _fused_hidden_rows
  //<< " _Input Matrix Threads : " << _input_matrix_threads << " Hidden Matrix Threads : " << _hidden_matrix_threads << std::endl;

#ifdef USE_MKL
  //Initialize weight matrixes here
  //malloc weights
  //_UZ, _UR, _UH have size _passge_input_dim * _hidden_dim
  //_WZ, _WR, _WH have size _hidden_dim * _hidden_dim
  _UZ = (float*)mkl_malloc(_input_dim * _hidden_dim * sizeof(float), 64);
  _UR = (float*)mkl_malloc(_input_dim * _hidden_dim * sizeof(float), 64);
  _UH = (float*)mkl_malloc(_input_dim * _hidden_dim * sizeof(float), 64);
  _WZ = (float*)mkl_malloc(_hidden_dim * _hidden_dim * sizeof(float), 64);
  _WR = (float*)mkl_malloc(_hidden_dim * _hidden_dim * sizeof(float), 64);
  _WH = (float*)mkl_malloc(_hidden_dim * _hidden_dim * sizeof(float), 64);

  /******************************************Test Matrices*******************************************/
  //_input = (float*)mkl_malloc(_input_dim*_input_batch_num*_MAX_INPUT_LENGTH * sizeof(float), 64);
  _outputZRH = (float*)mkl_malloc(_hidden_dim * 3 * _input_batch_num * _MAX_INPUT_LENGTH * sizeof(float), 64);
  _inputWeightsZRH = (float*)mkl_malloc(_input_dim * _hidden_dim * 3 * sizeof(float), 64);
  _hiddenWeightsZRH = (float*)mkl_malloc(_hidden_dim * _hidden_dim * 3 * sizeof(float), 64);
  _hiddenWeightsZR = (float*)mkl_malloc(_hidden_dim * _hidden_dim * 2 * sizeof(float), 64);

  _output0 = (float*)mkl_malloc(_hidden_dim * 3 * _input_batch_num * _MAX_INPUT_LENGTH * sizeof(float), 64);
  _output1 = (float*)mkl_malloc(_hidden_dim * 3 * _input_batch_num * _MAX_INPUT_LENGTH * sizeof(float), 64);
  _output2 = (float*)mkl_malloc(_hidden_dim * 3 * _input_batch_num * _MAX_INPUT_LENGTH * sizeof(float), 64);
  _output3 = (float*)mkl_malloc(_hidden_dim * 3 * _input_batch_num * _MAX_INPUT_LENGTH * sizeof(float), 64);
  _previous_state = (float*)mkl_malloc(_hidden_dim * _input_batch_num * sizeof(float), 64);

  /**************************************************************************************************/

  _hidden0 = (float*)mkl_malloc(_hidden_dim * sizeof(float), 64);
  _batched_hidden0 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _bias_Z = (float*)mkl_malloc(_hidden_dim * sizeof(float), 64);
  _bias_R = (float*)mkl_malloc(_hidden_dim * sizeof(float), 64);
  _bias_H = (float*)mkl_malloc(_hidden_dim * sizeof(float), 64);
  _batched_bias_Z = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _batched_bias_R = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _batched_bias_H = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);

  // Pre-allocate intermediate result buffers and reuse them (but many intermediate results do not need to be stored and we may only use a subset of them)
  // Initial intermediate results with either malloc or mkl_malloc
  _Z_t1 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _Z_t2 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);

  _R_t1 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _R_t2 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);

  _H_t1 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _H_t2 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _H_t3 = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);

  _Z = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _R = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);
  _H = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);

  if (!_is_fw) {
    _inputs_reverse = (float*)mkl_malloc(_MAX_INPUT_LENGTH * _input_batch_num * _input_dim * sizeof(float), 64);
    _outputs_reverse = (float*)mkl_malloc(_MAX_INPUT_LENGTH * _input_batch_num * _hidden_dim * sizeof(float), 64);
  }
#else
  //Initialize weight matrixes here
  //malloc weights
  //_UZ, _UR, _UH have size _passge_input_dim * _hidden_dim
  //_WZ, _WR, _WH have size _hidden_dim * _hidden_dim
  _UZ = new float[_input_dim * _hidden_dim];
  _UR = new float[_input_dim * _hidden_dim];
  _UH = new float[_input_dim * _hidden_dim];
  _WZ = new float[_hidden_dim * _hidden_dim];
  _WR = new float[_hidden_dim * _hidden_dim];
  _WH = new float[_hidden_dim * _hidden_dim];
  _hidden0 = new float[_hidden_dim];
  _batched_hidden0 = new float[_input_batch_num * _hidden_dim];
  //_batched_hidden_states = (float*)mkl_malloc((_MAX_INPUT_LENGTH + 1) * _input_batch_num * _hidden_dim * sizeof(float), 64); // We add one additional hidden state for the initial hidden0 state.
  //_batched_outputs = (float*)mkl_malloc(_MAX_INPUT_LENGTH * _input_batch_num * _hidden_dim * sizeof(float), 64);
  _bias_Z = new float[_hidden_dim];
  _bias_R = new float[_hidden_dim];
  _bias_H = new float[_hidden_dim];

  // Pre-allocate intermediate result buffers and reuse them
  _Z_t1 = new float[input_batch_num * hidden_dim];
  _Z_t2 = new float[input_batch_num * hidden_dim];

  _R_t1 = new float[input_batch_num * hidden_dim];
  _R_t2 = new float[input_batch_num * hidden_dim];

  _H_t1 = new float[input_batch_num * hidden_dim];
  _H_t2 = new float[input_batch_num * hidden_dim];
  _H_t3 = new float[input_batch_num * hidden_dim];

#endif
}

GRU::~GRU() {
#ifdef USE_MKL
  mkl_thread_free_buffers();

  if (!_is_fw) {
    mkl_free(_inputs_reverse);
    mkl_free(_outputs_reverse);
  }

  mkl_free(_Z_t1);
  mkl_free(_Z_t2);

  mkl_free(_R_t1);
  mkl_free(_R_t2);

  mkl_free(_H_t1);
  mkl_free(_H_t2);
  mkl_free(_H_t3);

  mkl_free(_Z);
  mkl_free(_R);
  mkl_free(_H);

  mkl_free(_UZ);
  mkl_free(_UR);
  mkl_free(_UH);
  mkl_free(_WZ);
  mkl_free(_WR);
  mkl_free(_WH);

  mkl_free(_outputZRH);
  mkl_free(_inputWeightsZRH);
  mkl_free(_hiddenWeightsZRH);
  mkl_free(_hiddenWeightsZR);
  mkl_free(_previous_state);

  mkl_free(_output0);
  mkl_free(_output1);
  mkl_free(_output2);
  mkl_free(_output3);

  mkl_free(_hidden0);
  mkl_free(_batched_hidden0);
  mkl_free(_bias_Z);
  mkl_free(_bias_R);
  mkl_free(_bias_H);
  mkl_free(_batched_bias_Z);
  mkl_free(_batched_bias_R);
  mkl_free(_batched_bias_H);
#else
  delete[] _Z;
  delete[] _R;
  delete[] _H;
  delete[] _Scur;
  delete[] _Snext;

  delete[] _Z_t1;
  delete[] _Z_t2;

  delete[] _R_t1;
  delete[] _R_t2;

  delete[] _H_t1;
  delete[] _H_t2;
  delete[] _H_t3;

  delete[] _UZ;
  delete[] _UR;
  delete[] _UH;
  delete[] _WZ;
  delete[] _WR;
  delete[] _WH;
  delete[] _hidden0;
  delete[] _batched_hidden0;
  delete[] _bias_Z;
  delete[] _bias_R;
  delete[] _bias_H;
  delete[] _batched_bias_Z;
  delete[] _batched_bias_R;
  delete[] _batched_bias_H;
#endif
}

void GRU::CopyWeightsToFusedWeights() {
  //copy the 4 indivudual input weight matrices into a single big input weight matrix
  for (int row = 0; row < _input_dim; row++)
    for (int col = 0; col < _hidden_dim; col++) {
      _inputWeightsZRH[row * _hidden_dim * 3 + 0 * _hidden_dim + col] = _UZ[row * _hidden_dim + col];
      _inputWeightsZRH[row * _hidden_dim * 3 + 1 * _hidden_dim + col] = _UR[row * _hidden_dim + col];
      _inputWeightsZRH[row * _hidden_dim * 3 + 2 * _hidden_dim + col] = _UH[row * _hidden_dim + col];
    }

  //copy the 4 individual hidden weight matrices into a single big hidden weight matrix
  for (int row = 0; row < _hidden_dim; row++)
    for (int col = 0; col < _hidden_dim; col++) {
      _hiddenWeightsZRH[row * _hidden_dim * 3 + 0 * _hidden_dim + col] = _WZ[row * _hidden_dim + col];
      _hiddenWeightsZRH[row * _hidden_dim * 3 + 1 * _hidden_dim + col] = _WR[row * _hidden_dim + col];
      _hiddenWeightsZRH[row * _hidden_dim * 3 + 2 * _hidden_dim + col] = _WH[row * _hidden_dim + col];

      _hiddenWeightsZR[row * _hidden_dim * 2 + 0 * _hidden_dim + col] = _WZ[row * _hidden_dim + col];
      _hiddenWeightsZR[row * _hidden_dim * 2 + 1 * _hidden_dim + col] = _WR[row * _hidden_dim + col];
    }
}
void GRU::AssignRandomWeights() {
  AssignRandomNumber(_UZ, _input_dim * _hidden_dim);
  AssignRandomNumber(_UR, _input_dim * _hidden_dim);
  AssignRandomNumber(_UH, _input_dim * _hidden_dim);
  AssignRandomNumber(_WZ, _hidden_dim * _hidden_dim);
  AssignRandomNumber(_WR, _hidden_dim * _hidden_dim);
  AssignRandomNumber(_WH, _hidden_dim * _hidden_dim);

  //copy the 4 indivudual input weight matrices into a single big input weight matrix
  /*for (int row = 0; row < _input_dim; row++)
        for (int col = 0; col < _hidden_dim; col++) {
            _inputWeightsZRH[row*_hidden_dim * 3 + 0 * _hidden_dim + col] = _UZ[row * _hidden_dim + col];
            _inputWeightsZRH[row*_hidden_dim * 3 + 1 * _hidden_dim + col] = _UR[row * _hidden_dim + col];
            _inputWeightsZRH[row*_hidden_dim * 3 + 2 * _hidden_dim + col] = _UH[row * _hidden_dim + col];
        }

    //copy the 4 individual hidden weight matrices into a single big hidden weight matrix
    for (int row = 0; row < _hidden_dim; row++)
        for (int col = 0; col < _hidden_dim; col++) {
            _hiddenWeightsZRH[row*_hidden_dim * 3 + 0 * _hidden_dim + col] = _WZ[row * _hidden_dim + col];
            _hiddenWeightsZRH[row*_hidden_dim * 3 + 1 * _hidden_dim + col] = _WR[row * _hidden_dim + col];
            _hiddenWeightsZRH[row*_hidden_dim * 3 + 2 * _hidden_dim + col] = _WH[row * _hidden_dim + col];


            _hiddenWeightsZR[row*_hidden_dim * 2 + 0 * _hidden_dim + col] = _WZ[row * _hidden_dim + col];
            _hiddenWeightsZR[row*_hidden_dim * 2 + 1 * _hidden_dim + col] = _WR[row * _hidden_dim + col];

        }*/

  CopyWeightsToFusedWeights();

  memset(_outputZRH, 0, _hidden_dim * 3 * _input_batch_num * _MAX_INPUT_LENGTH * sizeof(float));

  //AssignRandomNumber(_hidden0, _hidden_dim);
  // Set the initial hidden state to all 0s like TensorFlow does.
  memset(_hidden0, 0, sizeof(float) * _hidden_dim);
  // Replicate _batched_hidden0 with _input_batch_num X _hidden0
  RepeatVectorToConstructArray(_hidden0, _hidden_dim, _batched_hidden0, _input_batch_num);
  // By default, the gate bias are all 1s, and the candidate bias are all 0s (same as the GRU in TensorFlow).
  for (int i = 0; i < _hidden_dim; i++) {
    _bias_Z[i] = 1;
    _bias_R[i] = 1;
    _bias_H[i] = 0;
  }
  // Replicate the biases _input_batch_num copies to avoid module (%) operations in the gates calculation.
  RepeatVectorToConstructArray(_bias_Z, _hidden_dim, _batched_bias_Z, _input_batch_num);
  RepeatVectorToConstructArray(_bias_R, _hidden_dim, _batched_bias_R, _input_batch_num);
  RepeatVectorToConstructArray(_bias_H, _hidden_dim, _batched_bias_H, _input_batch_num);
#ifdef VERBOSE_GRU
  printf("GRU: batched hidden0:\n");
  PrintMatrix2D(_batched_hidden0, _input_batch_num, _hidden_dim);
#endif
}

void GRU::CopyWeightMatrix(float* UZ, float* WZ, float* UR, float* WR, float* UH, float* WH, float* hidden0) {
  // Copy weights into GRU pre-allocated buffers.
  ArrayCopy(UZ, _input_dim * _hidden_dim, _UZ);
  ArrayCopy(WZ, _hidden_dim * _hidden_dim, _WZ);
  ArrayCopy(UR, _input_dim * _hidden_dim, _UR);
  ArrayCopy(WR, _hidden_dim * _hidden_dim, _WR);
  ArrayCopy(UH, _input_dim * _hidden_dim, _UH);
  ArrayCopy(WH, _hidden_dim * _hidden_dim, _WH);
  ArrayCopy(hidden0, _hidden_dim, _hidden0);
  for (int i = 0; i < _input_batch_num; i++) {
    ArrayCopy(_hidden0, _hidden_dim, _batched_hidden0 + i * _hidden_dim);
  }

#ifdef VERBOSE_GRU
  printf("GRU: batched hidden0:\n");
  PrintMatrix2D(_batched_hidden0, _input_batch_num, _hidden_dim);
#endif

  CopyWeightsToFusedWeights();
}

void GRU::CopyWeightsBackToBack(const float* UZ_UR, const float* WZ_WR, const float* UH, const float* WH,
                                const float* bias_Z, const float* bias_R, const float* bias_H) {
  CopyMatrixBackToBack(UZ_UR, _UZ, _UR, _input_dim, _hidden_dim);
  CopyMatrixBackToBack(WZ_WR, _WZ, _WR, _hidden_dim, _hidden_dim);
  CopyMatrix(UH, _UH, _input_dim, _hidden_dim);
  CopyMatrix(WH, _WH, _hidden_dim, _hidden_dim);
  ArrayCopy((float*)bias_Z, _hidden_dim, _bias_Z);
  ArrayCopy((float*)bias_R, _hidden_dim, _bias_R);
  ArrayCopy((float*)bias_H, _hidden_dim, _bias_H);
  memset(_hidden0, 0, sizeof(float) * _hidden_dim);
  // Replicate _batched_hidden0 with _input_batch_num X _hidden0
  RepeatVectorToConstructArray(_hidden0, _hidden_dim, _batched_hidden0, _input_batch_num);
  // Replicate the biases _input_batch_num copies to avoid module (%) operations in the gates calculation.
  RepeatVectorToConstructArray(_bias_Z, _hidden_dim, _batched_bias_Z, _input_batch_num);
  RepeatVectorToConstructArray(_bias_R, _hidden_dim, _batched_bias_R, _input_batch_num);
  RepeatVectorToConstructArray(_bias_H, _hidden_dim, _batched_bias_H, _input_batch_num);

  CopyWeightsToFusedWeights();
}

void GRU::CopyWeights(const float* UZ, const float* WZ, const float* UR, const float* WR, const float* UH, const float* WH,
                      const float* bias_Z, const float* bias_R, const float* bias_H) {
  CopyMatrix(UZ, _UZ, _input_dim, _hidden_dim);
  CopyMatrix(WZ, _WZ, _hidden_dim, _hidden_dim);
  CopyMatrix(UR, _UR, _input_dim, _hidden_dim);
  CopyMatrix(WR, _WR, _hidden_dim, _hidden_dim);
  CopyMatrix(UH, _UH, _input_dim, _hidden_dim);
  CopyMatrix(WH, _WH, _hidden_dim, _hidden_dim);
  ArrayCopy((float*)bias_Z, _hidden_dim, _bias_Z);
  ArrayCopy((float*)bias_R, _hidden_dim, _bias_R);
  ArrayCopy((float*)bias_H, _hidden_dim, _bias_H);
  memset(_hidden0, 0, sizeof(float) * _hidden_dim);
  RepeatVectorToConstructArray(_hidden0, _hidden_dim, _batched_hidden0, _input_batch_num);
  int offset = 0;
  for (int i = 0; i < _input_batch_num; i++) {
    ArrayCopy((float*)bias_Z, _hidden_dim, _batched_bias_Z + offset);
    ArrayCopy((float*)bias_R, _hidden_dim, _batched_bias_R + offset);
    ArrayCopy((float*)bias_H, _hidden_dim, _batched_bias_H + offset);
    offset += _hidden_dim;
  }
}

void GRU::CopyWeightsWithTranspose(const float* UZ, const float* WZ, const float* UR, const float* WR, const float* UH, const float* WH,
                                   const float* bias_Z, const float* bias_R, const float* bias_H) {
  CopyMatrixWithTranspose(UZ, _UZ, _input_dim, _hidden_dim);
  CopyMatrixWithTranspose(WZ, _WZ, _hidden_dim, _hidden_dim);
  CopyMatrixWithTranspose(UR, _UR, _input_dim, _hidden_dim);
  CopyMatrixWithTranspose(WR, _WR, _hidden_dim, _hidden_dim);
  CopyMatrixWithTranspose(UH, _UH, _input_dim, _hidden_dim);
  CopyMatrixWithTranspose(WH, _WH, _hidden_dim, _hidden_dim);
  ArrayCopy((float*)bias_Z, _hidden_dim, _bias_Z);
  ArrayCopy((float*)bias_R, _hidden_dim, _bias_R);
  ArrayCopy((float*)bias_H, _hidden_dim, _bias_H);
  int offset = 0;
  for (int i = 0; i < _input_batch_num; i++) {
    ArrayCopy((float*)bias_Z, _hidden_dim, _batched_bias_Z + offset);
    ArrayCopy((float*)bias_R, _hidden_dim, _batched_bias_R + offset);
    ArrayCopy((float*)bias_H, _hidden_dim, _batched_bias_H + offset);
    offset += _hidden_dim;
  }

  CopyWeightsToFusedWeights();
}

void GRU::SaveWeights(std::string const& filename, std::string const& filename2) {
  std::ofstream out_file;
  // TODO: Handle corner cases such as filename being empty, etc.
  out_file.open(filename);

  int input_trans_matrix_size = _input_dim * _hidden_dim;
  int hidden_state_trans_matrix_size = _hidden_dim * _hidden_dim;

  SaveWeightsToFile(out_file, _UR, input_trans_matrix_size);
  SaveWeightsToFile(out_file, _WR, hidden_state_trans_matrix_size);
  SaveWeightsToFile(out_file, _UZ, input_trans_matrix_size);
  SaveWeightsToFile(out_file, _WZ, hidden_state_trans_matrix_size);
  SaveWeightsToFile(out_file, _UH, input_trans_matrix_size);
  SaveWeightsToFile(out_file, _WH, hidden_state_trans_matrix_size);
  SaveWeightsToFile(out_file, _bias_R, _hidden_dim);
  SaveWeightsToFile(out_file, _bias_Z, _hidden_dim);
  SaveWeightsToFile(out_file, _bias_H, _hidden_dim);
  //SaveWeightsToFile(out_file, _hidden0, _hidden_dim);

  out_file.close();
}

void GRU::RestoreWeights(std::string const& filename, std::string const& filename2) {
  std::ifstream in_file;
  in_file.open(filename);

  int input_trans_matrix_size = _input_dim * _hidden_dim;
  int hidden_state_trans_matrix_size = _hidden_dim * _hidden_dim;

  if (in_file.fail()) {
    std::cout << "File opening error" << std::endl;
  } else {
    RestoreWeightsToFile(in_file, _UR, input_trans_matrix_size);
    RestoreWeightsToFile(in_file, _WR, hidden_state_trans_matrix_size);
    RestoreWeightsToFile(in_file, _UZ, input_trans_matrix_size);
    RestoreWeightsToFile(in_file, _WZ, hidden_state_trans_matrix_size);
    RestoreWeightsToFile(in_file, _UH, input_trans_matrix_size);
    RestoreWeightsToFile(in_file, _WH, hidden_state_trans_matrix_size);
    RestoreWeightsToFile(in_file, _bias_R, _hidden_dim);
    RestoreWeightsToFile(in_file, _bias_Z, _hidden_dim);
    RestoreWeightsToFile(in_file, _bias_H, _hidden_dim);

    // Set the initial hidden state to all 0s like TensorFlow does.
    memset(_hidden0, 0, sizeof(float) * _hidden_dim);
    // Replicate _batched_hidden0 with _input_batch_num X _hidden0
    RepeatVectorToConstructArray(_hidden0, _hidden_dim, _batched_hidden0, _input_batch_num);

    // Replicate the biases _input_batch_num copies to avoid module (%) operations in the gates calculation.
    RepeatVectorToConstructArray(_bias_Z, _hidden_dim, _batched_bias_Z, _input_batch_num);
    RepeatVectorToConstructArray(_bias_R, _hidden_dim, _batched_bias_R, _input_batch_num);
    RepeatVectorToConstructArray(_bias_H, _hidden_dim, _batched_bias_H, _input_batch_num);

    CopyWeightsToFusedWeights();

#ifdef VERBOSE_GRU
    printf("bias_R:");
    PrintMatrix1D(_bias_R, _hidden_dim);
    printf("bias_Z:");
    PrintMatrix1D(_bias_Z, _hidden_dim);
    printf("bias_H:");
    PrintMatrix1D(_bias_H, _hidden_dim);
#endif

    in_file.close();
  }
}

void GRU::SaveInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths) {
  std::ofstream out_file;
  out_file.open(filename);

  if (out_file.fail()) {
    std::cout << "File opening error" << std::endl;
  } else {
    for (int i = 0; i < _MAX_INPUT_LENGTH; i++) {
      for (int j = 0; j < _input_batch_num; j++) {
        for (int k = 0; k < _input_dim; k++) {
          out_file << inputs[i * _input_batch_num * _input_dim + j * _input_dim + k] << " ";
        }
      }
    }
    out_file << std::endl;

    out_file.close();
  }
}

void GRU::SaveInputs(std::string const& filename, float* inputs) {
  std::ofstream out_file;
  out_file.open(filename);

  if (out_file.fail()) {
    std::cout << "File opening error" << std::endl;
  } else {
    for (int i = 0; i < _MAX_INPUT_LENGTH; i++) {
      for (int j = 0; j < _input_batch_num; j++) {
        for (int k = 0; k < _input_dim; k++) {
          out_file << inputs[i * _input_batch_num * _input_dim + j * _input_dim + k] << " ";
        }
      }
    }
    out_file << std::endl;

    out_file.close();
  }
}

void GRU::RestoreInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths) {
  std::ifstream in_file;
  in_file.open(filename);

  if (in_file.fail()) {
    std::cout << "File opening error" << std::endl;
  } else {
    // restore inputs (embedding)
    for (int i = 0; i < _MAX_INPUT_LENGTH; i++) {
      for (int j = 0; j < _input_batch_num; j++) {
        for (int k = 0; k < _input_dim; k++) {
          in_file >> inputs[i * _input_batch_num * _input_dim + j * _input_dim + k];
        }
      }
    }

    // restore sequence lengths
    for (int i = 0; i < _input_batch_num; i++) {
      in_file >> sequence_lengths[i];
    }

    in_file.close();
  }
}

void GRU::computeGRUsequence(float* input, int batch, int input_dim, int hidden_dim, int passage_length,
                             int version, int f_input_rows, int f_hidden_rows, int threads) {
  int Ni = batch;
  int Nj = hidden_dim;
  int Nk = input_dim;

  _input = input;

  //Input Matrix Part of GRU
  if (version == 0) {
    int fused_rows = f_input_rows;
    float alpha = 1.0f;
    float beta = 0.0f;

#pragma omp parallel for num_threads(threads)
    for (int st = 0; st < passage_length * Ni; st += fused_rows) {
      //std::cout << "St : "<< st << std::endl;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  fused_rows, Nj * 3, Nk, alpha,
                  _input + st * Nk, Nk,
                  _inputWeightsZRH, Nj * 3, beta,
                  _outputZRH + st * (Nj * 3), Nj * 3);
    }
  }

  //Hidden Matrix Part of GRU
  if (version == 1) {
    int fused_rows = f_hidden_rows;
    float alpha = 1.0f;
    float beta = 1.0f;

#pragma omp parallel for num_threads(threads)
    for (int bt = 0; bt < Ni; bt += fused_rows) {
      float* outAdded;
      float* previous_state = _previous_state + bt * Nj;
      for (int st = 0; st < passage_length; st++) {
        outAdded = _outputZRH + (st * Ni * Nj + bt * Nj) * 2;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    fused_rows, Nj * 2, Nj, alpha,
                    previous_state, Nj,
                    _hiddenWeightsZR, Nj * 2, beta,
                    outAdded, Nj * 3);

        //need to do the activation function computation
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    fused_rows, Nj, Nj, alpha,
                    outAdded, Nj,
                    _WR, Nj, beta,
                    outAdded + 2 * Nj, Nj * 3);

        //need to do the activation function computation
        previous_state = outAdded;
      }
    }
  }

  if (version == 2) {
    int fused_input_rows = f_input_rows;
    int fused_hidden_rows = f_hidden_rows;
    float alpha = 1.0f;
    float beta = 0.0f;

#pragma omp parallel for num_threads(threads)
    for (int st = 0; st < passage_length * Ni; st += fused_input_rows) {
      //std::cout << "St : "<< st << std::endl;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  fused_input_rows, Nj * 3, Nk, alpha,
                  _input + st * Nk, Nk,
                  _inputWeightsZRH, Nj * 3, beta,
                  _outputZRH + st * (Nj * 3), Nj * 3);
    }

    //std::cout << "Running version 4" << std::endl;
    alpha = 1.0f;
    beta = 1.0f;

#pragma omp parallel for num_threads(threads)
    for (int bt = 0; bt < Ni; bt += fused_hidden_rows) {
      float* outAdded;
      float* previous_state = _previous_state + bt * Nj;
      for (int st = 0; st < passage_length; st++) {
        outAdded = _outputZRH + (st * Ni * Nj + bt * Nj) * 2;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    fused_hidden_rows, Nj * 2, Nj, alpha,
                    previous_state, Nj,
                    _hiddenWeightsZR, Nj * 2, beta,
                    outAdded, Nj * 3);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    fused_hidden_rows, Nj, Nj, alpha,
                    outAdded, Nj,
                    _WR, Nj, beta,
                    outAdded + 2 * Nj, Nj * 3);
        //need to do the activation function computation
        previous_state = outAdded;
      }
    }
  }
}

void GRU::Compute(float* inputs, int max_steps, int32_t* sequence_lengths, float* outputs, float* final_states, bool isParallel) {
  // GRU Layer
  float* batched_input_one_step;
  float* batched_hidden_state_one_step = _batched_hidden0;
#ifdef VERBOSE_GRU
  printf("GRU: inputs from the ASR:\n");
  PrintMatrix3D(inputs, _MAX_INPUT_LENGTH, _input_batch_num, _input_dim);
  printf("GRU: initial hidden stats set by ASR:\n");
  PrintMatrix2D(batched_hidden_state_one_step, _input_batch_num, _hidden_dim);
#endif
  int input_step_length = _input_batch_num * _input_dim;
  int output_step_length = _input_batch_num * _hidden_dim;
  float* batched_output_one_step;  // [_input_batch_num, _hidden_dim]

  if (sequence_lengths == nullptr) {
    // If sequence lenghts are not passed
    sequence_lengths = CreateInt32Matrices(1, _input_batch_num);
    for (int i = 0; i < _input_batch_num; i++) {
      sequence_lengths[i] = _MAX_INPUT_LENGTH;
    }
  }

  float* output_buffer = outputs;
  if (!_is_fw) {
    ReverseSequence(inputs, _inputs_reverse, sequence_lengths, _MAX_INPUT_LENGTH, _input_batch_num, _input_dim);
    inputs = _inputs_reverse;
    outputs = _outputs_reverse;
#ifdef VERBOSE_GRU
    printf("reversed inputs:");
    PrintMatrix3D(inputs, _MAX_INPUT_LENGTH, _input_batch_num, _input_dim);
    std::ofstream out_file;
    out_file.open("./outputs/gru-bw-first-reverse-outputs-c++.txt");
    int output_len = _MAX_INPUT_LENGTH * _input_batch_num * _input_dim;
    for (int count = 0; count < output_len; count++) {
      out_file << _inputs_reverse[count] << " ";
    }
    out_file << std::endl;
    out_file.close();
#endif
  }

  // Calculate the max and min length
  int32_t max_sequence_length = 0;
  int32_t min_sequence_length = _MAX_INPUT_LENGTH;
  for (int i = 0; i < _input_batch_num; i++) {
    if (sequence_lengths[i] > max_sequence_length) {
      max_sequence_length = sequence_lengths[i];
    }
    if (sequence_lengths[i] < min_sequence_length) {
      min_sequence_length = sequence_lengths[i];
    }
  }

#ifdef VERBOSE_GRU
  printf("max_length: %d\n", max_sequence_length);
  printf("min_length: %d\n", min_sequence_length);
#endif

  int step = 0;
  while (step < max_steps) {
    batched_input_one_step = inputs + step * input_step_length;
    batched_output_one_step = outputs + step * output_step_length;

    if (step >= max_sequence_length) {
      for (int i = 0; i < _input_batch_num; i++) {
        memset(batched_output_one_step + i * _hidden_dim, 0, sizeof(float) * _hidden_dim);
      }
    } else {
      // Options: StepOnBatch (3 X 3), StepOnBatchParallelVer1Plus (2 X 3, 1, 1,1), StepOnBatchParallelVer2,
      if (isParallel) {
        StepOnBatchParallelVer7(batched_input_one_step, batched_hidden_state_one_step, batched_output_one_step);
      } else {
#ifdef VERBOSE_GRU_OUTPUT
        std::cout << std::endl
                  << " Step : " << step << std::endl
                  << std::endl;
        std::cout << " ------------------------------------------------------------" << std::endl;
#endif
        StepOnBatchSerial(batched_input_one_step, batched_hidden_state_one_step, batched_output_one_step);
      }

      if (step >= min_sequence_length) {
        for (int i = 0; i < _input_batch_num; i++) {
          if (step >= sequence_lengths[i]) {
            memset(batched_output_one_step + i * _hidden_dim, 0, sizeof(float) * _hidden_dim);
          }
        }
      }
    }

    batched_hidden_state_one_step = batched_output_one_step;
    step++;
  }

#ifdef VERBOSE_GRU_FINAL_OUTPUT
  printMatrix("Full Output ", outputs, _input_batch_num * max_sequence_length, _hidden_dim, 0, _hidden_dim);
#endif

#ifdef VERBOSE_GRU
  printf("gru output:");
  PrintMatrix3D(outputs, _MAX_INPUT_LENGTH, _input_batch_num, _hidden_dim);
#endif

  float* src;
  float* dest;
  int seq_len = 0;
  for (int i = 0; i < _input_batch_num; i++) {
    seq_len = sequence_lengths[i];
    if (seq_len <= 0) {
      break;
    }
    src = outputs + (seq_len - 1) * _input_batch_num * _hidden_dim + i * _hidden_dim;
    dest = final_states + i * _hidden_dim;
    std::memcpy(dest, src, sizeof(float) * _hidden_dim);
  }

  if (!_is_fw) {
    ReverseSequence(outputs, output_buffer, sequence_lengths, _MAX_INPUT_LENGTH, _input_batch_num, _hidden_dim);
#ifdef VERBOSE_GRU
    outputs = output_buffer;
    std::ofstream out_file;
    out_file.open("./outputs/gru-bw-second-reverse-outputs-c++.txt");
    int output_len = _MAX_INPUT_LENGTH * _input_batch_num * _hidden_dim;
    for (int count = 0; count < output_len; count++) {
      out_file << output_buffer[count] << " ";
    }
    out_file << std::endl;
    out_file.close();
#endif
  }
}

void GRU::ComputeV2(float* inputs, int max_steps, int32_t* sequence_lengths, float* outputs, float* final_states) {
  // GRU Layer
#ifdef VERBOSE_GRU
  float* batched_hidden_state_one_step = _batched_hidden0;
  printf("GRU: inputs from the ASR:\n");
  PrintMatrix3D(inputs, _MAX_INPUT_LENGTH, _input_batch_num, _input_dim);
  printf("GRU: initial hidden stats set by ASR:\n");
  PrintMatrix2D(batched_hidden_state_one_step, _input_batch_num, _hidden_dim);
#endif

  if (sequence_lengths == nullptr) {
    // If sequence lenghts are not passed
    sequence_lengths = CreateInt32Matrices(1, _input_batch_num);
    for (int i = 0; i < _input_batch_num; i++) {
      sequence_lengths[i] = _MAX_INPUT_LENGTH;
    }
  }

  float* output_buffer = outputs;
  if (!_is_fw) {
    ReverseSequence(inputs, _inputs_reverse, sequence_lengths, _MAX_INPUT_LENGTH, _input_batch_num, _input_dim);
    inputs = _inputs_reverse;
    outputs = _outputs_reverse;
#ifdef VERBOSE_GRU
    printf("reversed inputs:");
    PrintMatrix3D(inputs, _MAX_INPUT_LENGTH, _input_batch_num, _input_dim);
    std::ofstream out_file;
    out_file.open("./outputs/gru-bw-first-reverse-outputs-c++.txt");
    int output_len = _MAX_INPUT_LENGTH * _input_batch_num * _input_dim;
    for (int count = 0; count < output_len; count++) {
      out_file << _inputs_reverse[count] << " ";
    }
    out_file << std::endl;
    out_file.close();
#endif
  }

  // Calculate the max and min length
  int32_t max_sequence_length = 0;
  int32_t min_sequence_length = _MAX_INPUT_LENGTH;
  for (int i = 0; i < _input_batch_num; i++) {
    //std::cout << "Sequence Length at " << i << " is :" << sequence_lengths[i] << std::endl;
    if (sequence_lengths[i] > max_sequence_length) {
      max_sequence_length = sequence_lengths[i];
    }
    if (sequence_lengths[i] < min_sequence_length) {
      min_sequence_length = sequence_lengths[i];
    }
  }

#ifdef VERBOSE_GRU
  printf("max_length: %d\n", max_sequence_length);
  printf("min_length: %d\n", min_sequence_length);
#endif

  float alpha = 1.0f;
  float beta = 0.0f;

  int Ni = _input_batch_num;
  int Nj = _hidden_dim;
  int Nk = _input_dim;
  int Nj3 = 3 * Nj;
  int Nj2 = 2 * Nj;
  int total_rows = max_sequence_length * Ni;

  //////////////////////////////////////Core Calculations///////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  //Calculate the Input Matrix Multiplications in Parallel
#pragma omp parallel for num_threads(_input_matrix_threads)
  for (int row = 0; row < total_rows; row += _fused_input_rows) {
    //handling boundaries
    int local_fused_input_rows = _fused_input_rows;
    if ((row + _fused_input_rows) > total_rows)
      local_fused_input_rows = total_rows - row;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                local_fused_input_rows, Nj3, Nk, alpha,
                inputs + row * Nk, Nk,
                _inputWeightsZRH, Nj3, beta,
                _outputZRH + row * Nj3, Nj * 3);
  }

  //hidden weight MM results are added to _outputZRH
  alpha = 1.0f;
  beta = 1.0f;

  //Calculate the Hidden Matrix Multiplications in Parallel
#pragma omp parallel for num_threads(_hidden_matrix_threads)
  for (int row = 0; row < Ni; row += _fused_hidden_rows) {
    mkl_set_num_threads_local(1);
    //handling boundaries
    int local_fused_hidden_rows = _fused_hidden_rows;
    if ((row + _fused_hidden_rows) > Ni)
      local_fused_hidden_rows = Ni - row;

    float* outAdded;
    float* s_t1_prev = _batched_hidden0 + row * Nj;
    for (int step = 0; step < max_steps; step++) {
      if (step >= max_sequence_length) {
        for (int r = row; r < row + local_fused_hidden_rows; r++)
          for (int s = max_sequence_length; s < max_steps; s++)
            memset(outputs + s * Ni * Nj + r * Nj, 0, Nj * sizeof(float));
        break;
      }

#ifdef VERBOSE_GRU_OUTPUT
      std::cout << std::endl
                << " Step : " << step << std::endl
                << std::endl;
      std::cout << " ------------------------------------------------------------" << std::endl;
      printMatrix("Input ", inputs + (step * Ni + row) * Nk, local_fused_hidden_rows, _input_dim, 0, _input_dim);
      //printMatrix("InputWeightsIFOG", _inputWeightsIFOG, Nk, Nj4, 0, Nj4);
      printMatrix("Z_t1 ", _outputZRH + (step * Ni + row) * Nj3, local_fused_hidden_rows, _hidden_dim, 0, Nj3);
      printMatrix("R_t1 ", _outputZRH + (step * Ni + row) * Nj3 + Nj, local_fused_hidden_rows, _hidden_dim, 0, Nj3);
      printMatrix("H_t1 ", _outputZRH + (step * Ni + row) * Nj3 + 2 * Nj, local_fused_hidden_rows, _hidden_dim, 0, Nj3);

#endif
      outAdded = _outputZRH + (step * Ni + row) * Nj3;

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  local_fused_hidden_rows, Nj2, Nj, alpha,
                  s_t1_prev, Nj,
                  _hiddenWeightsZR, Nj2, beta,
                  outAdded, Nj3);

#ifdef VERBOSE_GRU_OUTPUT
      printMatrix("s_t1_prev ", s_t1_prev + row * _hidden_dim, local_fused_hidden_rows, _hidden_dim, 0, _hidden_dim);
      printMatrix("Z_t2 + Z_t1", outAdded, local_fused_hidden_rows, _hidden_dim, 0, Nj3);
      printMatrix("R_t2 + R_t2", outAdded + Nj, local_fused_hidden_rows, _hidden_dim, 0, Nj3);
#endif

      //1st Set Of Activations
      float* H_t2_local = _H_t2 + row * Nj;
      float* batched_bias_Z_local = _batched_bias_Z + row * Nj;
      float* batched_bias_R_local = _batched_bias_R + row * Nj;
      float* batched_bias_H_local = _batched_bias_H + row * Nj;

#if USE_AVX
      float* ps1 = outAdded + Nj;
      float* ps2 = batched_bias_R_local;
      float* ps4 = s_t1_prev;
      float* pd = H_t2_local;

      for (int r = 0; r < local_fused_hidden_rows; r++, ps1 += Nj3, ps2 += Nj, ps4 += Nj, pd += Nj) {
        RESET_GATE_V2(ps1, ps2, ps4, pd, Nj);
      }
#else
      int index = 0;
      for (int r = 0; r < local_fused_hidden_rows; r++) {
        for (int c = 0; c < Nj; c++) {
          H_t2_local[index] = s_t1_prev[index] * LUT::getInstance()->FastSigmoidUseLUT(outAdded[r * Nj3 + Nj + c] + batched_bias_R_local[index++]);  // bias[1]
        }
      }
      /*
         for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
         _H_t2[index] = s_t1_prev[index] * LUT::getInstance()->FastSigmoidUseLUT(_R_t1[index]
         + _R_t2[index] + _batched_bias_R[index]); // bias[1]
         }*/
#endif
      /*
         for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
         _H_t2[index] = s_t1_prev[index] * LUT::getInstance()->FastSigmoidUseLUT(_R_t1[index]
         + _R_t2[index] + _batched_bias_R[index]); // bias[1]
         }*/

#ifdef VERBOSE_GRU_OUTPUT

      printMatrix("Batched Bias Z ", batched_bias_Z_local, local_fused_hidden_rows, _hidden_dim, 0, _hidden_dim);
      printMatrix("Batched Bias H ", batched_bias_H_local, local_fused_hidden_rows, _hidden_dim, 0, _hidden_dim);
      printMatrix("Batched Bias R ", batched_bias_R_local, local_fused_hidden_rows, _hidden_dim, 0, _hidden_dim);

      printMatrix("H_t2 ", H_t2_local, local_fused_hidden_rows, _hidden_dim, 0, _hidden_dim);
#endif

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  local_fused_hidden_rows, Nj, Nj, alpha,
                  H_t2_local, Nj,
                  _WH, Nj, beta,
                  outAdded + Nj2, Nj3);

#ifdef VERBOSE_GRU_OUTPUT
      printMatrix("H_t3 + H_t1 ", outAdded + Nj2, local_fused_hidden_rows, _hidden_dim, 0, Nj3);
#endif
      //2nd Set of Activations
      float* output = outputs + (step * Ni + row) * Nj;

#if USE_AVX
      float* ps11 = outAdded;
      float* ps12 = batched_bias_Z_local;
      float* ps21 = outAdded + Nj2;
      float* ps22 = batched_bias_H_local;
      ps4 = s_t1_prev;
      pd = output;
      for (int r = 0; r < local_fused_hidden_rows; r++, ps11 += Nj3, ps12 += Nj, ps21 += Nj3, ps22 += Nj, ps4 += Nj, pd += Nj) {
        OUTPUT_GATE_V2(ps11, ps12, ps21, ps22, ps4, pd, Nj);
      }
#else
      float temp_z, temp_h;
      index = 0;
      for (int r = 0; r < local_fused_hidden_rows; r++) {
        for (int c = 0; c < Nj; c++) {
          temp_z = LUT::getInstance()->FastSigmoidUseLUT(outAdded[r * Nj3 + c] + batched_bias_Z_local[index]);
          temp_h = LUT::getInstance()->FastTanhUseLUT(outAdded[r * Nj3 + Nj2 + c] + batched_bias_H_local[index]);
          output[index] = (1 - temp_z) * temp_h + temp_z * s_t1_prev[index++];
        }
      }
#endif

      if (step >= min_sequence_length)
        for (int r = row; r < row + local_fused_hidden_rows; r++)
          if (step >= sequence_lengths[r])
            memset(outputs + (step * Ni + r) * Nj, 0, sizeof(float) * _hidden_dim);

#ifdef VERBOSE_GRU_OUTPUT
      printMatrix("Batched Output ", output, local_fused_hidden_rows, _hidden_dim, 0, _hidden_dim);
#endif

      s_t1_prev = output;
      /*
         float tmp_z;
         float tmp_h;
         for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
         tmp_z = LUT::getInstance()->FastSigmoidUseLUT(_Z_t1[index] + _Z_t2[index] + _batched_bias_Z[index]); //  bias[0]
         tmp_h = LUT::getInstance()->FastTanhUseLUT(_H_t1[index] + _H_t3[index] + _batched_bias_H[index]); // bias[2]																										  // Skip _Snext and store the hidden state outputs directly in batched_output since ASR does not need to convert _Snext to O through a softmax.
         batched_output[index] = (1 - tmp_z) * tmp_h + tmp_z * s_t1_prev[index];
         }
         */
    }
    mkl_set_num_threads_local(0);
  }

#ifdef VERBOSE_GRU_FINAL_OUTPUT
  printMatrix("Full Output ", outputs, _input_batch_num * max_sequence_length, _hidden_dim, 0, _hidden_dim);
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

#ifdef VERBOSE_GRU
  printf("gru output:");
  PrintMatrix3D(outputs, _MAX_INPUT_LENGTH, _input_batch_num, _hidden_dim);
#endif

  float* src;
  float* dest;
  int seq_len = 0;
  for (int i = 0; i < _input_batch_num; i++) {
    seq_len = sequence_lengths[i];

    src = outputs + (seq_len - 1) * _input_batch_num * _hidden_dim + i * _hidden_dim;
    dest = final_states + i * _hidden_dim;
    std::memcpy(dest, src, sizeof(float) * _hidden_dim);
  }

  if (!_is_fw) {
    ReverseSequence(outputs, output_buffer, sequence_lengths, _MAX_INPUT_LENGTH, _input_batch_num, _hidden_dim);
#ifdef VERBOSE_GRU
    outputs = output_buffer;
    std::ofstream out_file;
    out_file.open("./outputs/gru-bw-second-reverse-outputs-c++.txt");
    int output_len = _MAX_INPUT_LENGTH * _input_batch_num * _hidden_dim;
    for (int count = 0; count < output_len; count++) {
      out_file << output_buffer[count] << " ";
    }
    out_file << std::endl;
    out_file.close();
#endif
  }
}

void GRU::SetParallelism(int omp_thread, int mkl_thread) {
  num_omp_thread = omp_thread;
  num_mkl_thread = mkl_thread;
}

// Use 3 X 3 + 3 X 3 parallelism strategy.
//void GRU::StepOnBatch(float* batched_input, float* s_t1_prev, float* batched_output) {
//#pragma omp parallel num_threads(3)
//  {
//#pragma omp sections
//    {
//#pragma omp section
//      {
//        mkl_set_num_threads_local(3);
//        MatrixMult((float*)batched_input, (float*)_UR, (float*)_R_t1, _input_batch_num, _hidden_dim, _input_dim);
//      }
//
//#pragma omp section
//      {
//        mkl_set_num_threads_local(3);
//        MatrixMult((float*)s_t1_prev, (float*)_WR, (float*)_R_t2, _input_batch_num, _hidden_dim, _hidden_dim);
//      }
//
//#pragma omp section
//      {
//        mkl_set_num_threads_local(3);
//        MatrixMult((float*)batched_input, (float*)_UZ, (float*)_Z_t1, _input_batch_num, _hidden_dim, _input_dim);
//      }
//    }
//  }
//
//  int length_input_batch_multiply_hidden_dim = _input_batch_num * _hidden_dim;
//  int index;
//  for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
//    _H_t2[index] = s_t1_prev[index] * LUT::getInstance()->FastSigmoidUseLUT(_R_t1[index] + _R_t2[index]);
//  }
//
//#pragma omp parallel num_threads(3)
//  {
//#pragma omp sections
//    {
//#pragma omp section
//      {
//        mkl_set_num_threads_local(3);
//        MatrixMult((float*)s_t1_prev, (float*)_WZ, (float*)_Z_t2, _input_batch_num, _hidden_dim, _hidden_dim);
//      }
//
//#pragma omp section
//      {
//        mkl_set_num_threads_local(3);
//        MatrixMult((float*)_H_t2, (float*)_WH, (float*)_H_t3, _input_batch_num, _hidden_dim, _hidden_dim);
//      }
//
//#pragma omp section
//      {
//        mkl_set_num_threads_local(3);
//        MatrixMult((float*)batched_input, (float*)_UH, (float*)_H_t1, _input_batch_num, _hidden_dim, _input_dim);
//      }
//    }
//  }
//
//  float tmp_z;
//  float tmp_h;
//  for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
//    tmp_z = LUT::getInstance()->FastSigmoidUseLUT(_Z_t1[index] + _Z_t2[index]);
//    tmp_h = LUT::getInstance()->FastTanhUseLUT(_H_t1[index] + _H_t3[index]);
//    batched_output[index] = (1 - tmp_z) * tmp_h + tmp_z * s_t1_prev[index];
//  }
//}

void GRU::StepOnBatchSerial(float* batched_input, float* s_t1_prev, float* batched_output) {
#ifdef USE_MKL
  mkl_set_num_threads_local(1);
#endif
  // The implementation is based on the GRU cell step calculation at:https://arxiv.org/pdf/1406.1078v3.pdf (TF) and https://arxiv.org/pdf/1412.3555v1.pdf
  // Update gate
  MatrixMult((float*)batched_input, (float*)_UZ, (float*)_Z_t1, _input_batch_num, _hidden_dim, _input_dim);
  MatrixMult((float*)s_t1_prev, (float*)_WZ, (float*)_Z_t2, _input_batch_num, _hidden_dim, _hidden_dim);

#ifdef VERBOSE_GRU
  printf("GRU: batched inputs for the current step\n");
  PrintMatrix2D(batched_input, _input_batch_num, _input_dim);
  printf("GRU: batched hidden states from the previous step\n");
  PrintMatrix2D(s_t1_prev, _input_batch_num, _hidden_dim);
  printf("GRU: weights UZ\n");
  PrintMatrix2D(_UZ, _input_dim, _hidden_dim);
#endif

  // Reset gate
  MatrixMult((float*)batched_input, (float*)_UR, (float*)_R_t1, _input_batch_num, _hidden_dim, _input_dim);
  MatrixMult((float*)s_t1_prev, (float*)_WR, (float*)_R_t2, _input_batch_num, _hidden_dim, _hidden_dim);

  int length_input_batch_multiply_hidden_dim = _input_batch_num * _hidden_dim;
#if USE_AVX
  reset_gate_contfrac5_avx(_R_t1, _R_t2, _batched_bias_R, s_t1_prev, _H_t2, length_input_batch_multiply_hidden_dim);
#else
  int index;
  for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
    //_H_t2[index] = s_t1_prev[index] * LUT::getInstance()->FastSigmoidUseLUT(_R_t1[index] + _R_t2[index] + _batched_bias_R[index]); // bias[1]
    _H_t2[index] = s_t1_prev[index] * Sigmoid(_R_t1[index] + _R_t2[index] + _batched_bias_R[index]);  // bias[1]
  }
#endif

  MatrixMult((float*)batched_input, (float*)_UH, (float*)_H_t1, _input_batch_num, _hidden_dim, _input_dim);
  MatrixMult((float*)_H_t2, (float*)_WH, (float*)_H_t3, _input_batch_num, _hidden_dim, _hidden_dim);
#if USE_AVX
  output_gate_contfrac5_avx(_Z_t1, _Z_t2, _batched_bias_Z, _H_t1, _H_t3, _batched_bias_H, s_t1_prev, batched_output, length_input_batch_multiply_hidden_dim);
#else
  float tmp_z;
  float tmp_h;
  for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
    //tmp_z = LUT::getInstance()->FastSigmoidUseLUT(_Z_t1[index] + _Z_t2[index] + _batched_bias_Z[index]); //  bias[0]
    tmp_z = Sigmoid(_Z_t1[index] + _Z_t2[index] + _batched_bias_Z[index]);  //  bias[0]
    //tmp_h = LUT::getInstance()->FastTanhUseLUT(_H_t1[index] + _H_t3[index] + _batched_bias_H[index]); // bias[2]
    tmp_h = tanh(_H_t1[index] + _H_t3[index] + _batched_bias_H[index]);  // bias[2]
                                                                         // Skip _Snext and store the hidden state outputs directly in batched_output since ASR does not need to convert _Snext to O through a softmax.
    batched_output[index] = (1 - tmp_z) * tmp_h + tmp_z * s_t1_prev[index];
  }
#endif

#ifdef VERBOSE_GRU_OUTPUT
  printMatrix("Input ", batched_input, _input_batch_num, _input_dim, 0, _input_dim);
  //printMatrix("InputWeightsIFOG", _inputWeightsIFOG, Nk, Nj4, 0, Nj4);
  printMatrix("Z_t1 ", _Z_t1, _input_batch_num, _hidden_dim, 0, _hidden_dim);
  printMatrix("R_t1 ", _R_t1, _input_batch_num, _hidden_dim, 0, _hidden_dim);
  printMatrix("H_t1 ", _H_t1, _input_batch_num, _hidden_dim, 0, _hidden_dim);

  printMatrix("s_t1_prev ", s_t1_prev, _input_batch_num, _hidden_dim, 0, _hidden_dim);
  printMatrix("Z_t2 ", _Z_t2, _input_batch_num, _hidden_dim, 0, _hidden_dim);
  printMatrix("R_t2 ", _R_t2, _input_batch_num, _hidden_dim, 0, _hidden_dim);

  printMatrix("Batched Bias Z ", _batched_bias_Z, _input_batch_num, _hidden_dim, 0, _hidden_dim);
  printMatrix("Batched Bias H ", _batched_bias_H, _input_batch_num, _hidden_dim, 0, _hidden_dim);
  printMatrix("Batched Bias R ", _batched_bias_R, _input_batch_num, _hidden_dim, 0, _hidden_dim);

  printMatrix("H_t2 ", _H_t2, _input_batch_num, _hidden_dim, 0, _hidden_dim);
  printMatrix("H_t3 ", _H_t3, _input_batch_num, _hidden_dim, 0, _hidden_dim);

  printMatrix("Batched Output ", batched_output, _input_batch_num, _hidden_dim, 0, _hidden_dim);

#endif
#ifdef VERBOSE_GRU
  printf("GRU: batched output results\n");
  PrintMatrix2D(batched_output, _input_batch_num, _hidden_dim);
#endif
}

void GRU::StepOnBatchParallelVer7(float* batched_input, float* s_t1_prev, float* batched_output) {
  int length_input_batch_multiply_hidden_dim = _input_batch_num * _hidden_dim;
#pragma omp parallel sections num_threads(4)
  {
#pragma omp section
    {
      mkl_set_num_threads_local(3);
      MatrixMult((float*)s_t1_prev, (float*)_WZ, (float*)_Z_t2, _input_batch_num, _hidden_dim, _hidden_dim);
    }
#pragma omp section
    {
      mkl_set_num_threads_local(1);
      MatrixMult((float*)batched_input, (float*)_UZ, (float*)_Z_t1, _input_batch_num, _hidden_dim, _input_dim);
    }
#pragma omp section
    {
      mkl_set_num_threads_local(1);
      MatrixMult((float*)batched_input, (float*)_UH, (float*)_H_t1, _input_batch_num, _hidden_dim, _input_dim);
    }
#pragma omp section
    {
#pragma omp parallel sections num_threads(2)
      {
#pragma omp section
        {
          mkl_set_num_threads_local(5);
          MatrixMult((float*)s_t1_prev, (float*)_WR, (float*)_R_t2, _input_batch_num, _hidden_dim, _hidden_dim);
        }
#pragma omp section
        {
          mkl_set_num_threads_local(2);
          MatrixMult((float*)batched_input, (float*)_UR, (float*)_R_t1, _input_batch_num, _hidden_dim, _input_dim);
        }
      }
#if USE_AVX
      reset_gate_contfrac5_avx(_R_t1, _R_t2, _batched_bias_R, s_t1_prev, _H_t2, length_input_batch_multiply_hidden_dim);
#else
      //compute R (no need to store the intermediate results of R, which saves one read and write), and _H_t2
      for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
        _H_t2[index] = (float)(s_t1_prev[index] * LUT::getInstance()->FastSigmoidUseLUT((double)(_R_t1[index] + _R_t2[index] + _batched_bias_R[index])));
      }
#endif
      mkl_set_num_threads_local(7);
      MatrixMult((float*)_H_t2, (float*)_WH, (float*)_H_t3, _input_batch_num, _hidden_dim, _hidden_dim);
    }
  }

#if USE_AVX
  output_gate_contfrac5_avx(_Z_t1, _Z_t2, _batched_bias_Z, _H_t1, _H_t3, _batched_bias_H, s_t1_prev, batched_output, length_input_batch_multiply_hidden_dim);
#else
  //compute Z & H
  double tmp_z;
  double tmp_h;
  int index;
  for (index = 0; index < length_input_batch_multiply_hidden_dim; index++) {
    tmp_z = LUT::getInstance()->FastSigmoidUseLUT((double)(_Z_t1[index] + _Z_t2[index] + _batched_bias_Z[index]));
    tmp_h = LUT::getInstance()->FastTanhUseLUT((double)(_H_t1[index] + _H_t3[index] + _batched_bias_H[index]));
    // Skip _Snext and store the hidden state outputs directly in batched_output. This is because ASR does not need to convert _Snext to O through a softmax. If the GRU is used somewhere else, one might need to add the hidden_state->output transformation.
    batched_output[index] = (float)((1 - tmp_z) * tmp_h + tmp_z * s_t1_prev[index]);
  }
#endif
}

//kmp_affinity_mask_t socket_one_mask;
//kmp_affinity_mask_t socket_two_mask;
//
//struct MaskInit {
//  MaskInit(int num_proc) {
//    kmp_create_affinity_mask(&socket_one_mask);
//    kmp_create_affinity_mask(&socket_two_mask);
//    for (int i = 0; i < num_proc; i++) {
//      kmp_set_affinity_mask_proc(i, &socket_one_mask);
//      kmp_set_affinity_mask_proc(i + num_proc, &socket_two_mask);
//    }
//  }
//};
//MaskInit masks(24);

inline void BindToSocket() {
  //int level_one_id = omp_get_ancestor_thread_num(1);
  //if (level_one_id == 0) {
  //  if (kmp_set_affinity(&socket_one_mask) != 0) {
  //    printf("Error: kmp_set_affinity(sizeof(socket_one_mask), &socket_one_mask)\n");
  //  }
  //} else {
  //  if (kmp_set_affinity(&socket_two_mask) != 0) {
  //    printf("Error: kmp_set_affinity(sizeof(socket_one_mask), &socket_one_mask)\n");
  //  }
  //}
}

void BindLevelTwoThreads() {
  //int level_one_id = omp_get_ancestor_thread_num(1);
  //int level_two_id = omp_get_ancestor_thread_num(2);

  //int assigned_id = level_one_id * 24 + level_two_id * 2;

  //forceAffinity(assigned_id);
}

void BindLevelThreeThreads() {
  //int level_one_id = omp_get_ancestor_thread_num(1);
  //int level_two_id = omp_get_ancestor_thread_num(2);
  //int level_three_id = omp_get_ancestor_thread_num(3);

  //int assigned_id = level_one_id * 24 + level_two_id * 2 + 2 + level_three_id * 2;

  //forceAffinity(assigned_id);
}

// Assume the inputs is in the shape of [max_steps, n_batches, n_inputs].
void GRU::ReverseSequence(float* inputs, float* inputs_reverse, int32_t* sequence_lengths, int32_t max_sequence_length, int32_t input_batch_num, int32_t input_dim) {
  int seq_len = 0;
  float* src;
  float* dest;
  for (int i = 0; i < input_batch_num; i++) {
    seq_len = sequence_lengths[i];
    if (seq_len == 0) {
      continue;
    }
    for (int j = 0; j < seq_len; j++) {
      src = inputs + j * input_batch_num * input_dim + i * input_dim;
      dest = inputs_reverse + (seq_len - j - 1) * input_batch_num * input_dim + i * input_dim;
      std::memcpy(dest, src, sizeof(float) * input_dim);
    }
    for (int j = seq_len; j < max_sequence_length; j++) {
      src = inputs + j * input_batch_num * input_dim + i * input_dim;
      dest = inputs_reverse + j * input_batch_num * input_dim + i * input_dim;
      std::memcpy(dest, src, sizeof(float) * input_dim);
    }
  }
}

inline void GRU::setNumThreads(int input_matrix_threads, int hidden_matrix_threads) {
  _input_matrix_threads = input_matrix_threads;

  _fused_input_rows = _input_batch_num * _MAX_INPUT_LENGTH / _input_matrix_threads;
  if (_input_batch_num * _MAX_INPUT_LENGTH % _input_matrix_threads != 0)
    _fused_input_rows++;

  _hidden_matrix_threads = hidden_matrix_threads;

  _fused_hidden_rows = _input_batch_num / _hidden_matrix_threads;
  if (_input_batch_num % _hidden_matrix_threads != 0)
    _fused_hidden_rows++;

#ifdef VERBOSE_GRU
  std::cout << std::endl
            << "Manually Setting GRU Thread Numbers to (" << input_matrix_threads << ", " << hidden_matrix_threads << ")" << std::endl;
  std::cout << "Fused Input Rows : " << _fused_input_rows << " Fused Hidden Rows : " << _fused_hidden_rows
            << " _Input Matrix Threads : " << _input_matrix_threads << " Hidden Matrix Threads : " << _hidden_matrix_threads << std::endl;
#endif
}

inline void GRU::setFusedRows() {
  for (int t = MAX_NUM_CORES; t >= 1; t--) {
    if (_input_batch_num * _MAX_INPUT_LENGTH % t == 0) {
      _input_matrix_threads = t;
      _fused_input_rows = _input_batch_num * _MAX_INPUT_LENGTH / t;
      break;
    }
  }

  for (int t = MAX_NUM_CORES; t >= 1; t--) {
    if (_input_batch_num % t == 0) {
      _hidden_matrix_threads = t;
      _fused_hidden_rows = _input_batch_num / t;
      break;
    }
  }

  if (_input_matrix_threads < 8 || _hidden_matrix_threads < 8) {
    setNumThreads(12, 12);
#ifdef VERBOSE_GRU
    std::cout << std::endl
              << "Low number of GRU threads (Input Matrix Threads, Hidden Matrix Threads) : (" << _input_matrix_threads << ", " << _hidden_matrix_threads << ")" << std::endl;
    std::cout << "Consider setting threads manually using setNumThreads(..., ...), to improve performance." << std::endl;
    std::cout << "Ignore this message if setNumThreads is called in future." << std::endl;
#endif
  }
#ifdef VERBOSE_GRU
  std::cout << "Fused Input Rows : " << _fused_input_rows << " Fused Hidden Rows : " << _fused_hidden_rows
            << " _Input Matrix Threads : " << _input_matrix_threads << " Hidden Matrix Threads : " << _hidden_matrix_threads << std::endl;
#endif
}

const float* GRU::getCellState() {
  return NULL;
}

inline void GRU::setThreadsPerSocket(int threads) {
  _threads = threads;
}

/*void GRU::AutoTune(int max_threads_per_socket, int min_threads_per_socket) {
	int num_arrays = 1000;
	int num_iterations = 1000;
	double min_time = 999999.0;

	int input_threads = max_threads_per_socket;
	int hidden_threads = min_threads_per_socket;

	float* inputs = (float*)mkl_malloc(num_arrays* _MAX_INPUT_LENGTH * _input_batch_num * _input_dim * sizeof(float), 64);
	AssignRandomNumber(inputs, _MAX_INPUT_LENGTH * _input_batch_num * _input_dim);

	int* sequence_length = new int[_input_batch_num];
	for (int i = 0; i < _input_batch_num; i++) {
		sequence_length[i] = _MAX_INPUT_LENGTH;
	}

	float* outputs = (float*)mkl_malloc(_MAX_INPUT_LENGTH * _input_batch_num * _hidden_dim * sizeof(float), 64);
	float* final_states = (float*)mkl_malloc(_input_batch_num* _hidden_dim * sizeof(float), 64);

	// Start testing
	Stopwatch watch;
	watch.Restart();
	for (int ithreads = max_threads_per_socket; ithreads >= min_threads_per_socket; ithreads--) {
		for (int hthreads = max_threads_per_socket; hthreads >= min_threads_per_socket; hthreads--) {
			setNumThreads(ithreads, hthreads);
			watch.Restart();
			for (int i = 0; i < num_iterations; i++) {
				int index = rand() % (num_arrays - 1);
				ComputeV2(inputs + index * _MAX_INPUT_LENGTH * _input_batch_num* _input_dim, _MAX_INPUT_LENGTH, sequence_length, outputs, final_states);
			}
			watch.Stop();
			double total_seconds = watch.GetTimeInSeconds();
			std::cout << "Input Threads : " << ithreads << " Hidden Threads : " << hthreads << " Total Time : " << total_seconds << std::endl;
			if (total_seconds < min_time) {
				min_time = total_seconds;
				input_threads = ithreads;
				hidden_threads = hthreads;
			}
		}
	}
	setNumThreads(input_threads, hidden_threads);
	mkl_thread_free_buffers();
	mkl_free(inputs);
	mkl_free(outputs);
	mkl_free(final_states);
	delete[] sequence_length;


}*/
