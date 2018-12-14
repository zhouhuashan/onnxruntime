//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: gru.h
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

//#define VERBOSE_GRU
#define PARALLEL_GRU

/*Uncomment to get additional output. WARNING run a 
small problem size to avoid massive IO to the display*/
#define GRU_CORRECTNESS_TEST 1

//#define VERBOSE_GRU_OUTPUT 1
//#define VERBOSE_GRU_FINAL_OUTPUT 1

#include "RNN_fs.h"

class GRU final : public RNN {
 public:
  FASTASRSERVINGLIBRARY_API GRU(int input_dim, int input_batch_num, int hidden_dim, int MAX_INPUT_LENGTH, bool is_fw);
  FASTASRSERVINGLIBRARY_API void RestoreWeights(std::string const& filename, std::string const& filename2 = "");
  FASTASRSERVINGLIBRARY_API void SaveWeights(std::string const& filename, std::string const& filename2 = "");
  FASTASRSERVINGLIBRARY_API void SaveInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths);
  FASTASRSERVINGLIBRARY_API void SaveInputs(std::string const& filename, float* inputs);
  FASTASRSERVINGLIBRARY_API void RestoreInputs(std::string const& filename, float* inputs, int32_t* sequence_lengths);
  FASTASRSERVINGLIBRARY_API void AssignRandomWeights();
  void CopyWeightMatrix(float* UZ, float* WZ, float* UR, float* WR, float* UH, float* WH, float* hidden0);
  //void AssignInitialHiddenState(float* hidden0);
  FASTASRSERVINGLIBRARY_API void CopyWeightsBackToBack(const float* UZ_UR, const float* WZ_WR, const float* UH, const float* WH,
                                                       const float* bias_Z, const float* bias_R, const float* bias_H);
  FASTASRSERVINGLIBRARY_API void CopyWeights(const float* UZ, const float* WZ, const float* UR, const float* WR, const float* UH, const float* WH,
                                             const float* bias_Z, const float* bias_R, const float* bias_H);
  FASTASRSERVINGLIBRARY_API void CopyWeightsWithTranspose(const float* UZ, const float* WZ, const float* UR, const float* WR, const float* UH, const float* WH,
                                                          const float* bias_Z, const float* bias_R, const float* bias_H);
  FASTASRSERVINGLIBRARY_API void computeGRUsequence(float* input, int batch, int input_dim, int hidden_dim,
                                                    int passage_length, int version, int f_input_rows, int f_hidden_rows, int threads);

  /*This method must be called to copy the weights from individual weights
	matrices to the fused weight matrices.*/
  FASTASRSERVINGLIBRARY_API void CopyWeightsToFusedWeights();

  //Compute: take one or a batch of inputs to produce ouputs of all steps for all inputs
  //inputs : input_length x input_batch_num x input_dim
  //outputs: input_length x input_batch_number x hidden_dim
  FASTASRSERVINGLIBRARY_API void Compute(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states, bool isParallel);  //what is the format of inputs and outputs
  //FASTASRSERVINGLIBRARY_API void ComputeV3(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states, bool isParallel);  //what is the format of inputs and outputs
  FASTASRSERVINGLIBRARY_API void ComputeV2(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states);
  //FASTASRSERVINGLIBRARY_API void ComputeV2HardSigmoid(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states, bool isParallel);
  FASTASRSERVINGLIBRARY_API void setNumThreads(int input_matrix_threads, int hidden_matrix_threads);

  //FASTASRSERVINGLIBRARY_API void StepOnBatch(float* batched_input, float* s_t1_prev, float* batched_output);
  void StepOnBatchSerial(float* batched_input, float* s_t1_prev, float* batched_output);
  void SetParallelism(int num_omp_thread, int num_mkl_thread);
  void ReverseSequence(float* inputs, float* _inputs_reverse, int32_t* sequence_lengths, int32_t max_sequence_length, int32_t input_batch_num, int32_t input_dim);
  FASTASRSERVINGLIBRARY_API virtual const float* getCellState();
  FASTASRSERVINGLIBRARY_API ~GRU();

 private:
  /*int _input_dim;
	int _input_batch_num;
	int _MAX_INPUT_LENGTH;
	//int input_length; //could change per input
	int _hidden_dim;
	bool _is_fw;
	*/
  float* _inputs_reverse;
  float* _outputs_reverse;

  /*int _input_matrix_threads;
	int _hidden_matrix_threads;
	int _fused_input_rows;
	int _fused_hidden_rows;*/
  int _threads;

  //definition below corresponds to the formula here: http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
  //weights
  //_UZ, _UR, _UH have size _input_dim * _hidden_dim
  //_WZ, _WR, _WH have size _hidden_dim * _hidden_dim
  float* _UZ;
  float* _WZ;
  float* _UR;
  float* _WR;
  float* _UH;
  float* _WH;

  ////////////////////////Compute V2 Variables/////////////////////////////////////////
  float *_inputWeightsZRH, *_hiddenWeightsZRH, *_hiddenWeightsZR;

  float* _outputZRH;

  float* _input;
  float *_output0, *_output1, *_output2, *_output3, *_outAdded, *_previous_state;
  ////////////////////////////////////////////////////////////////////////////////

  //initial hidden state
  float* _hidden0;  //1 * hidden_dim
  float* _batched_hidden0;

  // Initial biases
  float* _bias_Z;
  float* _bias_R;
  float* _bias_H;

  float* _batched_bias_Z;
  float* _batched_bias_R;
  float* _batched_bias_H;

  //intermediate result buffers
  //below are vectors of size _input_batch_num
  float* _Z;
  float* _R;
  float* _H;
  float* _Scur;
  float* _Snext;

  //we also need a few intermediate buffers of size _input_batch_num x hidden_dim to hold computation results inside GRUCell.
  float* _Z_t1;
  float* _Z_t2;

  float* _R_t1;
  float* _R_t2;

  float* _H_t1;
  float* _H_t2;
  float* _H_t3;

  // Control inter-op and intra-op parallelism
  /*int num_omp_thread;
	int num_mkl_thread;*/
  //private functions

  //void StepOnBatchParallelVer1(float* batched_input, float* s_t1_prev, float* batched_output);
  //void StepOnBatchParallelVer2(float* batched_input, float* s_t1_prev, float* batched_output);
  //void StepOnBatchParallelVer3(float* batched_input, float* s_t1_prev, float* batched_output);
  //void StepOnBatchParallelVer4(float* batched_input, float* s_t1_prev, float* batched_output);
  //void StepOnBatchParallelVer5(float* batched_input, float* s_t1_prev, float* batched_output);
  //void StepOnBatchParallelVer6(float* batched_input, float* s_t1_prev, float* batched_output);
  void StepOnBatchParallelVer7(float* batched_input, float* s_t1_prev, float* batched_output);
  //void StepOnBatchParallelVer8(float* batched_input, float* s_t1_prev, float* batched_output);
  //void StepOnBatchParallelVer1WithBinding(float* batched_input, float* s_t1_prev, float* batched_output);
  //void ComputeHalfBatch(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states, int whichHalf);

  inline void setFusedRows();
  inline void setThreadsPerSocket(int threads);

  // Drop-in replacement of TensorFlow GRU cell computation.
  void GRUBlockCellFprop(int batch_size, int input_size, int cell_size, float* batched_input, float* h_prev, float* input_weight_reset, float* input_weight_update, float* input_weight_connection, float* recurrent_weight_reset, float* recurrent_weight_update, float* recurrent_weight_connection, float* bias_reset, float* bias_update, float* bias_connection, float* temp_u1, float* temp_u2, float* temp_r1, float* temp_r2, float* temp_c1, float* temp_c2, float* temp_c3, float* output_reset, float* output_update, float* output_connection, float* batched_output);
  void GRUBlockCellFpropOpenMp(int batch_size, int input_size, int cell_size, float* batched_input, float* h_prev, float* input_weight_reset, float* input_weight_update, float* input_weight_connection, float* recurrent_weight_reset, float* recurrent_weight_update, float* recurrent_weight_connection, float* bias_reset, float* bias_update, float* bias_connection, float* temp_u1, float* temp_u2, float* temp_r1, float* temp_r2, float* temp_c1, float* temp_c2, float* temp_c3, float* output_reset, float* output_update, float* output_connection, float* batched_output);
  // Parallel version for unidirectional encoding (1 passage encoding + 1 question encoding, which might be used for distributed serving)
  //void StepOnBatchParallelVer2(float* batched_input, float* s_t1_prev, float* batched_output);
};
