//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: BidirectionalRNN.h
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
#include "RNN_fs.h"
#include <mkl.h>
#include "gru_fs.h"
#include <memory>

// Make this class abstract as well. The constructor contains sub-class specific parameters.
class BidirectionalRNN : public RNN {
 public:
  FASTASRSERVINGLIBRARY_API BidirectionalRNN(int input_dim, int input_batch_num, int hidden_dim, int MAX_INPUT_LENGTH, RNNCell rnn_cell_type, float forget_gate = 0.0);
  FASTASRSERVINGLIBRARY_API void RestoreWeights(std::string const& filename_fw, std::string const& filename_bw);
  FASTASRSERVINGLIBRARY_API void SaveWeights(std::string const& filename_fw, std::string const& filename_bw);

  FASTASRSERVINGLIBRARY_API void LoadTFWeights(std::string const& filename, std::string const& filename2);
  FASTASRSERVINGLIBRARY_API void LoadTFBias(std::string const& filename, std::string const& filename2);

  FASTASRSERVINGLIBRARY_API void SaveInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths) {}
  FASTASRSERVINGLIBRARY_API void SaveInputs(std::string const& filename, float* inputs) {}

  FASTASRSERVINGLIBRARY_API void RestoreInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths) {}

  FASTASRSERVINGLIBRARY_API void AssignRandomWeights();

  /*This method must be called to copy the weights from individual weights
    matrices to the fused weight matrices.*/
  FASTASRSERVINGLIBRARY_API void CopyWeightsToFusedWeights();

  FASTASRSERVINGLIBRARY_API void Compute(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states, bool is_parallel);  //what is the format of inputs and outputs
  FASTASRSERVINGLIBRARY_API void ComputeV2(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states);
  FASTASRSERVINGLIBRARY_API void ComputeV2Transpose(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states);

  FASTASRSERVINGLIBRARY_API void setNumThreads(int input_matrix_threads, int hidden_matrix_threads);
  FASTASRSERVINGLIBRARY_API int getNumOutputElements();
  FASTASRSERVINGLIBRARY_API void AutoTune(int max_num_threads_per_socket, int min_num_threads_per_socket);
  FASTASRSERVINGLIBRARY_API ~BidirectionalRNN();
  FASTASRSERVINGLIBRARY_API const float* getCellState();

 private:
  RNN* _forwardRNN;
  RNN* _backwardRNN;
  float *_fw_out_buffer, *_bw_out_buffer;
  std::shared_ptr<float> _cellState;
  float *_fw_final_states_buffer, *_bw_final_states_buffer;
  void MergeForwardBackwardOutputs(float* outputs, float* final_states);
  void MergeForwardBackwardOutputsAndTranspose(float* outputs, float* final_states);
};
