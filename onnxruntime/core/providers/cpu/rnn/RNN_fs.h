//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: RNN.h
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#pragma once

#include <iterator>
#ifdef FASTASRSERVINGLIBRARY_EXPORTS
#define FASTASRSERVINGLIBRARY_API
#else
#define FASTASRSERVINGLIBRARY_API
#endif

#include <iostream>
#include <string>
#include <iomanip>

enum RNNCell { LSTMCell,
               GRUCell,
               BidirectionalLSTMCell,
               BidirectionalGRUCell,
               BaseCell };

class RNN {
 public:
  FASTASRSERVINGLIBRARY_API RNN(int input_dim, int input_batch_num, int hidden_dim, int MAX_INPUT_LENGTH, bool is_fw = false)
      : _input_dim(input_dim),
        _input_batch_num(input_batch_num),
        _hidden_dim(hidden_dim),
        _MAX_INPUT_LENGTH(MAX_INPUT_LENGTH),
        _is_fw(is_fw) {}

  FASTASRSERVINGLIBRARY_API virtual void RestoreWeights(std::string const& filename, std::string const& filename2 = "") = 0;
  FASTASRSERVINGLIBRARY_API virtual void SaveWeights(std::string const& filename, std::string const& filename2 = "") = 0;
  FASTASRSERVINGLIBRARY_API virtual void SaveInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths) = 0;
  FASTASRSERVINGLIBRARY_API virtual void SaveInputs(std::string const& filename, float* inputs) = 0;
  FASTASRSERVINGLIBRARY_API virtual void RestoreInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths) = 0;
  FASTASRSERVINGLIBRARY_API virtual void AssignRandomWeights() = 0;

  //Load TensorFlow Weights and Bias
  FASTASRSERVINGLIBRARY_API virtual void LoadTFWeights(std::string const& filename, std::string const& filename2 = "") {}
  FASTASRSERVINGLIBRARY_API virtual void LoadTFBias(std::string const& filename, std::string const& filename2 = "") {}

  /*This method must be called to copy the weights from individual weights
    matrices to the fused weight matrices.*/
  FASTASRSERVINGLIBRARY_API virtual void CopyWeightsToFusedWeights() = 0;

  FASTASRSERVINGLIBRARY_API virtual void Compute(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states, bool is_parallel) = 0;
  FASTASRSERVINGLIBRARY_API virtual void ComputeV2(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states) = 0;

  FASTASRSERVINGLIBRARY_API virtual void setNumThreads(int input_matrix_threads, int hidden_matrix_threads) = 0;
  FASTASRSERVINGLIBRARY_API int getInputMatrixThreads() { return _input_matrix_threads; }
  FASTASRSERVINGLIBRARY_API int getHiddenMatrixThreads() { return _hidden_matrix_threads; }

  FASTASRSERVINGLIBRARY_API virtual int getNumOutputElements() {
    return _hidden_dim * _input_batch_num * _MAX_INPUT_LENGTH;
  }

  FASTASRSERVINGLIBRARY_API virtual void AutoTune(int max_threads_per_socket, int min_threads_per_socket);

  FASTASRSERVINGLIBRARY_API RNNCell getRNNType() { return _cell_type; }

  FASTASRSERVINGLIBRARY_API virtual ~RNN() {}
  FASTASRSERVINGLIBRARY_API virtual const float* getCellState() = 0;

 protected:
  int _input_dim;
  int _input_batch_num;
  int _MAX_INPUT_LENGTH;
  int _hidden_dim;
  bool _is_fw;

  // Control inter-op and intra-op parallelism
  int num_omp_thread;
  int num_mkl_thread;

  //Threading Parameters
  int _input_matrix_threads;
  int _hidden_matrix_threads;
  int _fused_input_rows;
  int _fused_hidden_rows;

  RNNCell _cell_type;
  FASTASRSERVINGLIBRARY_API virtual void setRNNType(RNNCell cell_type) { _cell_type = BaseCell; }
};
