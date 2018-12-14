//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: lstm.cpp
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "RNN_fs.h"
#include <fstream>
#include <assert.h>
#include "low_level_func.h"
//#include "StopWatch.h"

void RNN::AutoTune(int /*max_threads_per_socket*/, int /*min_threads_per_socket*/) {
  //int num_arrays = 1000;
  //int num_iterations = 100;
  //double min_time = 999999.0;

  //int input_threads = max_threads_per_socket;
  //int hidden_threads = min_threads_per_socket;

  //float* inputs = (float*)mkl_malloc(num_arrays * _MAX_INPUT_LENGTH * _input_batch_num * _input_dim * sizeof(float), 64);
  //AssignRandomNumber(inputs, _MAX_INPUT_LENGTH * _input_batch_num * _input_dim);

  //int* sequence_length = new int[_input_batch_num];
  //for (int i = 0; i < _input_batch_num; i++) {
  //  sequence_length[i] = _MAX_INPUT_LENGTH;
  //}

  //float* outputs = (float*)mkl_malloc(_MAX_INPUT_LENGTH * _input_batch_num * _hidden_dim * sizeof(float), 64);
  //float* final_states = (float*)mkl_malloc(_input_batch_num * _hidden_dim * sizeof(float), 64);

  //// Start testing
  ////Stopwatch watch;
  ////watch.Restart();
  //for (int ithreads = max_threads_per_socket; ithreads >= min_threads_per_socket; ithreads--) {
  //  for (int hthreads = max_threads_per_socket; hthreads >= min_threads_per_socket; hthreads--) {
  //    setNumThreads(ithreads, hthreads);
  //    //watch.Restart();
  //    for (int i = 0; i < num_iterations; i++) {
  //      int index = rand() % (num_arrays - 1);
  //      ComputeV2(inputs + index * _MAX_INPUT_LENGTH * _input_batch_num * _input_dim, _MAX_INPUT_LENGTH, sequence_length, outputs, final_states);
  //    }
  //    //watch.Stop();
  //    //double total_seconds = watch.GetTimeInSeconds();
  //    //std::cout << "Input Threads : " << ithreads << " Hidden Threads : " << hthreads << " Total Time : " << total_seconds << std::endl;
  //    //if (total_seconds < min_time) {
  //    //  min_time = total_seconds;
  //    //  input_threads = ithreads;
  //    //  hidden_threads = hthreads;
  //    //}
  //  }
  //}
  //setNumThreads(input_threads, hidden_threads);
  //mkl_thread_free_buffers();
  //mkl_free(inputs);
  //mkl_free(outputs);
  //mkl_free(final_states);
  //delete[] sequence_length;

  //setNumThreads(max_threads_per_socket, max_threads_per_socket);
}
