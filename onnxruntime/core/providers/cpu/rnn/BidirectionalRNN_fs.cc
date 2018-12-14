//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: BidirectionalRNN.cpp
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#include "BidirectionalRNN_fs.h"
#include "InnerUtility.h"

BidirectionalRNN::BidirectionalRNN(int input_dim, int input_batch_num, int hidden_dim, int MAX_INPUT_LENGTH, RNNCell rnn_cell_type, float forget_gate) : RNN(input_dim, input_batch_num, hidden_dim, MAX_INPUT_LENGTH) {
  //RNNCell rnn_cell_type = GRUCell;
  if (rnn_cell_type == GRUCell) {
    _forwardRNN = new GRU(input_dim, input_batch_num, hidden_dim, MAX_INPUT_LENGTH, true);
    _backwardRNN = new GRU(input_dim, input_batch_num, hidden_dim, MAX_INPUT_LENGTH, false);
    _cell_type = BidirectionalGRUCell;

  } /*else if (rnn_cell_type == LSTMCell) {
    _forwardRNN = new LSTM(input_dim, input_batch_num, hidden_dim, MAX_INPUT_LENGTH, true, forget_gate);
    _backwardRNN = new LSTM(input_dim, input_batch_num, hidden_dim, MAX_INPUT_LENGTH, false, forget_gate);
    _cell_type = BidirectionalLSTMCell;
    _cellState = MemAllocate<float>(hidden_dim * 2 * input_batch_num);
  }*/
  else {
    std::cerr << "Unrecognized Cell Type. Exiting !!!";
    return;
  }

  _fw_out_buffer = (float*)mkl_malloc(_input_batch_num * hidden_dim * MAX_INPUT_LENGTH * sizeof(float), 64);
  _bw_out_buffer = (float*)mkl_malloc(_input_batch_num * hidden_dim * MAX_INPUT_LENGTH * sizeof(float), 64);
  _fw_final_states_buffer = (float*)mkl_malloc(_input_batch_num * hidden_dim * sizeof(float), 64);
  _bw_final_states_buffer = (float*)mkl_malloc(_input_batch_num * hidden_dim * sizeof(float), 64);
}

void BidirectionalRNN::RestoreWeights(std::string const& filename_fw, std::string const& filename_bw) {
  _forwardRNN->RestoreWeights(filename_fw);
  _backwardRNN->RestoreWeights(filename_bw);
}

void BidirectionalRNN::SaveWeights(std::string const& filename_fw, std::string const& filename_bw) {
  _forwardRNN->SaveWeights(filename_fw);
  _backwardRNN->SaveWeights(filename_bw);
}

void BidirectionalRNN::LoadTFWeights(std::string const& filename_fw, std::string const& filename_bw) {
  _forwardRNN->LoadTFWeights(filename_fw);
  _backwardRNN->LoadTFWeights(filename_bw);
}

void BidirectionalRNN::LoadTFBias(std::string const& filename_fw, std::string const& filename_bw) {
  _forwardRNN->LoadTFBias(filename_fw);
  _backwardRNN->LoadTFBias(filename_bw);
}

void BidirectionalRNN::AssignRandomWeights() {
  _forwardRNN->AssignRandomWeights();
  _backwardRNN->AssignRandomWeights();
}

/*This method must be called to copy the weights from individual weights
matrices to the fused weight matrices.*/
void BidirectionalRNN::CopyWeightsToFusedWeights() {
  _forwardRNN->CopyWeightsToFusedWeights();
  _backwardRNN->CopyWeightsToFusedWeights();
}

void BidirectionalRNN::Compute(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states, bool is_parallel) {
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
      _forwardRNN->Compute(inputs, input_length, sequence_lengths, _fw_out_buffer, _fw_final_states_buffer, is_parallel);
    }
#pragma omp section
    {
      _backwardRNN->Compute(inputs, input_length, sequence_lengths, _bw_out_buffer, _bw_final_states_buffer, is_parallel);
    }
  }
  MergeForwardBackwardOutputs(outputs, final_states);
}

void BidirectionalRNN::ComputeV2(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states) {
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
      _forwardRNN->ComputeV2(inputs, input_length, sequence_lengths, _fw_out_buffer, _fw_final_states_buffer);
    }
#pragma omp section
    {
      _backwardRNN->ComputeV2(inputs, input_length, sequence_lengths, _bw_out_buffer, _bw_final_states_buffer);
    }
  }
  MergeForwardBackwardOutputs(outputs, final_states);
}

void BidirectionalRNN::ComputeV2Transpose(float* inputs, int input_length, int32_t* sequence_lengths, float* outputs, float* final_states) {
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
      _forwardRNN->ComputeV2(inputs, input_length, sequence_lengths, _fw_out_buffer, _fw_final_states_buffer);
    }
#pragma omp section
    {
      _backwardRNN->ComputeV2(inputs, input_length, sequence_lengths, _bw_out_buffer, _bw_final_states_buffer);
    }
  }
  MergeForwardBackwardOutputsAndTranspose(outputs, final_states);
}

void BidirectionalRNN::MergeForwardBackwardOutputs(float* outputs, float* final_states) {
  //TODO  : Look up how to create a blocked schedule with OpenMP
  //#pragma omp parallel for
  for (int row = 0; row < _input_batch_num * _MAX_INPUT_LENGTH; row++) {
    memcpy(outputs + row * 2 * _hidden_dim, _fw_out_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
    memcpy(outputs + (row * 2 + 1) * _hidden_dim, _bw_out_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
  }

  //TODO  : Look up how to create a blocked schedule with OpenMP
  //#pragma omp parallel for
  for (int row = 0; row < _input_batch_num; row++) {
    memcpy(final_states + row * 2 * _hidden_dim, _fw_final_states_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
    memcpy(final_states + (row * 2 + 1) * _hidden_dim, _bw_final_states_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
  }
}

void BidirectionalRNN::MergeForwardBackwardOutputsAndTranspose(float* outputs, float* final_states) {
  //TODO  : Look up how to create a blocked schedule with OpenMP
  //#pragma omp parallel for
  for (int row = 0; row < _input_batch_num * _MAX_INPUT_LENGTH; row++) {
    int batch = row % _input_batch_num;
    int seq = row / _MAX_INPUT_LENGTH;
    memcpy(outputs + (batch * _MAX_INPUT_LENGTH + seq) * 2 * _hidden_dim, _fw_out_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
    memcpy(outputs + ((batch * _MAX_INPUT_LENGTH + seq) * 2 + 1) * _hidden_dim, _bw_out_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
  }

  //#pragma omp parallel for
  for (int row = 0; row < _input_batch_num; row++) {
    memcpy(final_states + row * 2 * _hidden_dim, _fw_final_states_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
    memcpy(final_states + (row * 2 + 1) * _hidden_dim, _bw_final_states_buffer + row * _hidden_dim, _hidden_dim * sizeof(float));
  }
}

void BidirectionalRNN::setNumThreads(int input_matrix_threads, int hidden_matrix_threads) {
  _forwardRNN->setNumThreads(input_matrix_threads, hidden_matrix_threads);
  _backwardRNN->setNumThreads(input_matrix_threads, hidden_matrix_threads);

  _input_matrix_threads = input_matrix_threads;
  _hidden_matrix_threads = hidden_matrix_threads;

  _fused_input_rows = _input_batch_num * _MAX_INPUT_LENGTH / _input_matrix_threads;
  if (_input_batch_num * _MAX_INPUT_LENGTH % _input_matrix_threads != 0)
    _fused_input_rows++;

  _fused_hidden_rows = _input_batch_num / _hidden_matrix_threads;
  if (_input_batch_num % _hidden_matrix_threads != 0)
    _fused_hidden_rows++;
}

void BidirectionalRNN::AutoTune(int max_num_threads_per_socket, int min_num_threads_per_socket) {
  _forwardRNN->AutoTune(max_num_threads_per_socket, min_num_threads_per_socket);
  int input_matrix_threads = _forwardRNN->getInputMatrixThreads();
  int hidden_matrix_threads = _forwardRNN->getHiddenMatrixThreads();
  setNumThreads(input_matrix_threads, hidden_matrix_threads);
}

int BidirectionalRNN::getNumOutputElements() {
  return 2 * _hidden_dim * _input_batch_num * _MAX_INPUT_LENGTH;
}

const float* BidirectionalRNN::getCellState() {
  if (this->_cell_type == BidirectionalLSTMCell) {
    auto fwCellState = this->_forwardRNN->getCellState();
    if (fwCellState == NULL) {
      return NULL;
    }
    auto bwCellState = this->_backwardRNN->getCellState();
    if (bwCellState == NULL) {
      return NULL;
    }
    float* output = this->_cellState.get();

    for (int i = 0; i < this->_input_batch_num; i++) {
      memcpy(output, fwCellState, sizeof(float) * _hidden_dim);
      output += _hidden_dim;
      fwCellState += _hidden_dim;

      memcpy(output, bwCellState, sizeof(float) * _hidden_dim);
      output += _hidden_dim;
      bwCellState += _hidden_dim;
    }
    return this->_cellState.get();
  }
  return NULL;
}

BidirectionalRNN::~BidirectionalRNN() {
  mkl_free(_fw_out_buffer);
  mkl_free(_fw_final_states_buffer);
  mkl_free(_bw_out_buffer);
  mkl_free(_bw_final_states_buffer);
  delete (_forwardRNN);
  delete (_backwardRNN);
}
