#include "uni_dir_attn_lstm.h"

using namespace onnxruntime::Rnn::detail;

namespace onnxruntime {
namespace ml {
namespace rnn {
namespace detail {

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::Rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

template <typename T>
UniDirectionalAttnLstm<T>::UniDirectionalAttnLstm(AllocatorPtr allocator,
                                                  const Logging::Logger& logger,
                                                  const int seq_length,
                                                  const int batch_size,
                                                  const int input_size,
                                                  const int hidden_size,
                                                  Direction direction,
                                                  const bool input_forget,
                                                  AttentionWrapper<T>& attention_wrapper,
                                                  const gsl::span<const T>& input_weights,
                                                  const gsl::span<const T>& recurrent_weights,
                                                  const gsl::span<const T>& bias,
                                                  const gsl::span<const T>& peephole_weights,
                                                  const gsl::span<const T>& initial_hidden_state,
                                                  const gsl::span<const T>& initial_cell_state,
                                                  const ActivationFuncs::Entry& activation_func_f,
                                                  const ActivationFuncs::Entry& activation_func_g,
                                                  const ActivationFuncs::Entry& activation_func_h,
                                                  const float clip,
                                                  TaskThreadPool& ttp)
    : allocator_(allocator),
      logger_(logger),
      seq_length_(seq_length),
      batch_size_(batch_size),
      input_size_(input_size),
      hidden_size_(hidden_size),
      direction_(direction),
      input_forget_(input_forget),
      attention_wrapper_(attention_wrapper),
      clip_(clip),
      use_bias_(!bias.empty()),
      use_peepholes_(!peephole_weights.empty()),
      ttp_(ttp) {
  activation_f_ = {deepcpu::ActivationFuncByName(activation_func_f.name),
                   activation_func_f.alpha,
                   activation_func_f.beta};

  activation_g_ = {deepcpu::ActivationFuncByName(activation_func_g.name),
                   activation_func_g.alpha,
                   activation_func_g.beta};

  activation_h_ = {deepcpu::LstmMergeGatesFuncByName(activation_func_h.name),
                   activation_func_h.alpha,
                   activation_func_h.beta};

  clip_with_bias_ptr_ = use_bias_ ? deepcpu::clip_add_bias : deepcpu::clip_ignore_bias;

  attention_size_ = attention_wrapper_.GetAttentionSize();
  attention_context_size_ = attention_wrapper_.GetAttentionContextSize();

  SetNumThreads();
  AllocateBuffers();
  InitializeBuffers(initial_hidden_state, initial_cell_state);
  LoadAllWeights(input_weights, recurrent_weights, peephole_weights, bias);
}

template <typename T>
void UniDirectionalAttnLstm<T>::AllocateBuffers() {
  weights_ifoc_ = Allocate(allocator_, input_size_ * hidden_size_ * 4, weights_ifoc_ptr);
  recurrent_weights_ifoc_ = Allocate(allocator_, hidden_size_ * hidden_size_ * 4, recurrent_weights_ifoc_ptr_);

  weights_attn_ifoc_ = Allocate(allocator_, attention_size_ * hidden_size_ * 4, weights_attn_ifoc_ptr_);

  // allocate and fill with 0's.
  const bool fill = true;
  hidden0_ = Allocate(allocator_, hidden_size_, hidden0_ptr_, fill);
  internal_memory_prev_ = Allocate(allocator_, hidden_size_, internal_memory_prev_ptr_, fill);
  internal_memory_cur_ = Allocate(allocator_, hidden_size_, internal_memory_cur_ptr_, fill);
  batched_hidden0_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_hidden0_ptr_, fill);

  batched_internal_memory_prev_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                           batched_internal_memory_prev_ptr_, fill);
  batched_internal_memory_cur_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                          batched_internal_memory_cur_ptr_, fill);
  batched_internal_memory_clipped_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                              batched_internal_memory_clipped_ptr_, fill);

  output_ifoc_ = Allocate(allocator_, hidden_size_ * 4 * batch_size_ * seq_length_, output_ifoc_ptr_, fill);

  if (use_bias_) {
    bias_WRi_ = Allocate(allocator_, hidden_size_, bias_WRi_ptr_);
    bias_WRf_ = Allocate(allocator_, hidden_size_, bias_WRf_ptr_);
    bias_WRo_ = Allocate(allocator_, hidden_size_, bias_WRo_ptr_);
    bias_WRc_ = Allocate(allocator_, hidden_size_, bias_WRc_ptr_);
  }

  if (direction_ == kReverse) {
    inputs_reverse_ = Allocate(allocator_, seq_length_ * batch_size_ * input_size_, inputs_reverse_ptr_);
    outputs_reverse_ = Allocate(allocator_, seq_length_ * batch_size_ * hidden_size_, outputs_reverse_ptr_);
  }

#if !defined(LSTM_NO_PEEPHOLE_COPY)
  if (use_peepholes_) {
    peephole_i_ = Allocate(allocator_, hidden_size_, peephole_i_ptr_);
    peephole_f_ = Allocate(allocator_, hidden_size_, peephole_f_ptr_);
    peephole_o_ = Allocate(allocator_, hidden_size_, peephole_o_ptr_);
  }
#endif
}

template <typename T>
void UniDirectionalAttnLstm<T>::InitializeBuffers(const gsl::span<const T>& initial_hidden_state,
                                                  const gsl::span<const T>& initial_cell_state) {
  if (!initial_hidden_state.empty()) {
    std::copy(initial_hidden_state.cbegin(), initial_hidden_state.cend(), batched_hidden0_.begin());
  } else {
    std::fill(batched_hidden0_.begin(), batched_hidden0_.end(), T{});
  }

  if (!initial_cell_state.empty()) {
    std::copy(initial_cell_state.cbegin(), initial_cell_state.cend(), batched_internal_memory_prev_.begin());
  } else {
    std::fill(batched_internal_memory_prev_.begin(), batched_internal_memory_prev_.end(), T{});
  }
}

template <typename T>
void UniDirectionalAttnLstm<T>::LoadAllWeights(const gsl::span<const T>& input_weights,
                                               const gsl::span<const T>& recurrent_weights,
                                               const gsl::span<const T>& peephole_weights,
                                               const gsl::span<const T>& bias) {
  DumpMatrix("W[iofc]_Transposed", input_weights.data(), 4 * hidden_size_, (attention_size_ + input_size_));
  LoadWeightsWithTranspose(input_weights, weights_ifoc_, hidden_size_, input_size_, input_size_ + attention_size_, 0);
  DumpMatrix("W[ifoc]", weights_ifoc_.data(), input_size_, 4*hidden_size_);
  LoadWeightsWithTranspose(input_weights, weights_attn_ifoc_, hidden_size_, attention_size_, input_size_ + attention_size_, input_size_);
  DumpMatrix("W[ifoc]_Attn", weights_attn_ifoc_.data(), attention_size_, 4 * hidden_size_);

  DumpMatrix("R[iofc]_Transposed", recurrent_weights.data(), 4 * hidden_size_, hidden_size_);
  LoadWeightsWithTranspose(recurrent_weights, recurrent_weights_ifoc_, hidden_size_, hidden_size_);
  DumpMatrix("R[ifoc]", recurrent_weights_ifoc_.data(), hidden_size_, 4 * hidden_size_);

  if (!peephole_weights.empty())
    LoadPeepholeWeights(peephole_weights);

  if (!bias.empty())
    LoadBias(bias);
}

// Load weights and transpose
template <typename T>
void UniDirectionalAttnLstm<T>::LoadWeightsWithTranspose(const gsl::span<const T>& input_weights,
                                                         gsl::span<T>& output_weights,
                                                         int dim0_size, int dim1_size, int stride1, int offset1) {
  const int i_in = 0;
  const int o_in = 1;
  const int f_in = 2;
  const int c_in = 3;

  const int i_out = 0;
  const int f_out = 1;
  const int o_out = 2;
  const int c_out = 3;

  if (stride1 < 0) stride1 = dim1_size;

  const int weight_size = dim0_size * stride1;
  const int fused_offset = 4 * dim0_size;

  // process each weight type in a separate thread. we could use more or less threads as a refinement
  // but starting with the simple approach first. provides ~2x speedup to cntk.test_HTK_LSTM_Truncated_Distributed
  // over the original code when did each copy/transpose sequentially.
  std::vector<std::future<void>> task_results{};
  task_results.reserve(4);

  auto copy_weights_with_transpose = [&](int out, int in) {
    auto out_offset = out * dim0_size;
    for (int row = 0; row < dim1_size; row++) {
      auto in_offset = in * weight_size + offset1 + row;
      for (int c = 0; c < dim0_size; c++) {
        // original code but much slower due to the calculations on each loop.
        // output_weights[row * fused_offset + out * dim0_size + c] = input_weights[in * weight_size + c * stride1 + offset1 + row];
        output_weights[out_offset + c] = input_weights[in_offset];
        in_offset += stride1;
      }
      out_offset += fused_offset;
    }
  };

  auto add_task = [&](int out, int in) {
    std::packaged_task<void()> task{std::bind(copy_weights_with_transpose, out, in)};
    task_results.push_back(task.get_future());
    ttp_.RunTask(std::move(task));
  };

  add_task(i_out, i_in);
  add_task(f_out, f_in);
  add_task(o_out, o_in);
  add_task(c_out, c_in);

  try {
    // wait for all and propagate any exceptions
    for (auto& future : task_results)
      future.get();
  } catch (const std::exception& ex) {
    LOGS(logger_, ERROR) << "Loading weights - exception running tasks: " << ex.what();
    throw;
  }
}

template <typename T>
void UniDirectionalAttnLstm<T>::LoadPeepholeWeights(const gsl::span<const T>& peephole_weights) {
  int i = 0;
#if defined(LSTM_NO_PEEPHOLE_COPY)
  // just use spans. we don't change these values so there's no point copying to them
  peephole_i_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);
  peephole_o_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);
  peephole_f_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);

#else
  DumpMatrix("P[i]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("P[o]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("P[f]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);

  auto copy_weight = [this, &peephole_weights](int offset, gsl::span<T>& out) {
    typename gsl::span<const T>::const_iterator in_iter = peephole_weights.cbegin() + offset;
    std::copy(in_iter, in_iter + hidden_size_, out.begin());
  };

  i = 0;
  copy_weight((i++ * hidden_size_), peephole_i_);
  copy_weight((i++ * hidden_size_), peephole_o_);
  copy_weight((i++ * hidden_size_), peephole_f_);
#endif

  /*
  DumpMatrix("peephole_i_", peephole_i_.data(), 1, hidden_size_);
  DumpMatrix("peephole_o_", peephole_o_.data(), 1, hidden_size_);
  DumpMatrix("peephole_f_", peephole_f_.data(), 1, hidden_size_);
  */
}

template <typename T>
void UniDirectionalAttnLstm<T>::LoadBias(const gsl::span<const T>& WbRb_values) {
  // add Wb and Rb
  auto copy_fused_bias = [this, &WbRb_values](int offset, gsl::span<T>& out) {
    // gap between Wb and Wb value for an entry
    const int Wb_to_Rb_offset = 4 * hidden_size_;
    for (int j = 0; j < hidden_size_; ++j) {
      out[j] = WbRb_values[j + offset] + WbRb_values[j + offset + Wb_to_Rb_offset];
    }
  };

  int i = 0;
  copy_fused_bias((i++) * hidden_size_, bias_WRi_);
  copy_fused_bias((i++) * hidden_size_, bias_WRo_);
  copy_fused_bias((i++) * hidden_size_, bias_WRf_);
  copy_fused_bias((i++) * hidden_size_, bias_WRc_);
}

template <typename T>
void UniDirectionalAttnLstm<T>::Compute(const gsl::span<const T>& inputs_arg,
                                        const gsl::span<const int>& sequence_lengths_arg,
                                        const int num_directions,
                                        gsl::span<T>& outputs,
                                        gsl::span<T>& final_hidden_state,
                                        gsl::span<T>& final_cell_state) {
  // copy spans (just T* and size, not data in span) as we may change them
  gsl::span<const T> inputs = inputs_arg;
  gsl::span<const int> sequence_lengths = sequence_lengths_arg;

  // if sequence lengths weren't provided, use internal array and init all to seq_length
  if (sequence_lengths.empty()) {
    sequence_lengths_ = Allocate(allocator_, batch_size_, sequence_lengths_ptr_, true, seq_length_);
    sequence_lengths = sequence_lengths_;
  }

  // LSTM Layer
  gsl::span<T> batched_hidden_state_one_step = batched_hidden0_;
  gsl::span<T> batched_internal_state_prev_one_step = batched_internal_memory_prev_;
  gsl::span<T> batched_internal_state_clipped_one_step = batched_internal_memory_clipped_;

  int output_step_length = batch_size_ * hidden_size_;

  // The bidirectional LSTM wrapper wraps this LSTM class and produces bi-directional output
  // the output has layout [seq,num_direction,batch,neurons].
  // When num_direction is 2, then this class will compute forward or backward LSTM.
  // The outputs corresponds to either [seq,0,batch,neurons] or [seq,1,batch,neurons]
  // Setting output_step_length this way allows writing the output directly without requiring
  // additional memcpy. Note that if direction is kReverse, we write to output_reverse buffer
  // which is then copied to output buffer, and ReverseSequence method handles the step length.
  if (direction_ == Direction::kForward && num_directions == 2)
    output_step_length = 2 * batch_size_ * hidden_size_;

  gsl::span<T> original_outputs = outputs;
  const bool output_sequence = !outputs.empty();

  if (direction_ == Direction::kReverse) {
    ReverseSequence(inputs, inputs_reverse_, sequence_lengths, seq_length_, batch_size_, input_size_, 1);
    inputs = inputs_reverse_;

    if (output_sequence)
      outputs = outputs_reverse_;
  }

  // Calculate the max and min length
  int32_t max_sequence_length = *std::max_element(sequence_lengths.cbegin(), sequence_lengths.cend());

  // macro 'min' coming from windows headers breaks 'std::min'
#undef min
  int32_t min_sequence_length = std::min(seq_length_, *std::min_element(sequence_lengths.cbegin(),
                                                                        sequence_lengths.cend()));

  ///**************************LSTM Calculations****************************/
  const int hidden_size_x4 = 4 * hidden_size_;
  const int total_rows = max_sequence_length * batch_size_;

#if defined(HAVE_PARALLELIZED_GEMM)
  // apply the weights to all the inputs and save to output_IFOC
  ComputeGemm(total_rows, hidden_size_x4, input_size_, T{1.0},
              inputs.cbegin(), inputs.cend(),
              input_size_,
              weights_ifoc_.cbegin(), weights_ifoc_.cend(),  // W[ifoc]^T
              hidden_size_x4, T{0.0},
              output_ifoc_.begin(), output_ifoc_.end(),
              hidden_size_x4);
#else

  int fused_input_rows = total_rows / input_num_threads_;
  if (total_rows % input_num_threads_ != 0)
    fused_input_rows++;

  // apply the weights to all the inputs and save to output_IFOC
  auto input_gemm = [&](int row) {
    //handling boundaries
    int local_fused_input_rows = fused_input_rows;
    if ((row + fused_input_rows) > total_rows)
      local_fused_input_rows = total_rows - row;

    // compute Xt*(W[ifoc]^T)
    ComputeGemm(local_fused_input_rows, hidden_size_x4, input_size_, T{1.0},
                inputs.cbegin() + row * input_size_, inputs.cend(),  // Xt
                input_size_,
                weights_ifoc_.cbegin(), weights_ifoc_.cend(),  // W[ifoc]^T
                hidden_size_x4, T{0.0},
                output_ifoc_.begin() + row * hidden_size_x4, output_ifoc_.end(),
                hidden_size_x4);
  };

  ExecuteLambdaInParallel("Applying weights to inputs", input_gemm, total_rows, fused_input_rows,
                          ttp_, logger_);
#endif

  DumpMatrix("Xt*(W[ifoc]^T)", output_ifoc_.data(), total_rows, hidden_size_x4);

  int fused_hidden_rows = batch_size_ / hidden_num_threads_;
  if (batch_size_ % hidden_num_threads_ != 0)
    fused_hidden_rows++;

  // NOTE: we could refine the bounds checking in the calls below that use these values to instead
  // explicitly check just the range for each iteration, however if it's going to run over
  // it should also run over on the last iteration, so this should be good enough to catch any
  // logic errors causing bounds violations.
  span_T_iter C_prev_end = batched_internal_state_prev_one_step.end();
  span_T_iter C_prev_clipped_end = batched_internal_state_clipped_one_step.end();
  span_T_const_iter previous_state_end = batched_hidden_state_one_step.end();

  {
    span_T_iter c_prev = batched_internal_state_prev_one_step.begin();
    span_T_iter c_prev_clipped = batched_internal_state_clipped_one_step.begin();

    // hidden state can be provided as input for first step, so need to special case that.
    // after the first step this will switch to the output from the previous step
    span_T_const_iter previous_state = batched_hidden_state_one_step.cbegin();

    //run through steps sequentially
    for (int step = 0; step < max_sequence_length; step++) {
      const std::string seqno_str = " [seqno=" + std::to_string(step) + "]";

      DumpMatrix("previous_state" + seqno_str, &*previous_state, batch_size_, hidden_size_);

      span_T_iter step_out_IFOC = output_ifoc_.begin() + (step * batch_size_) * hidden_size_x4;

    // shape is [ attention_size_ ]
      const gsl::span<const T> attention = attention_wrapper_.GetAttnStates();

    #if defined(HAVE_PARALLELIZED_GEMM)
      // Xt*(W[ifoc]^T) = INPUTt * W[ifoc]^T + At-1 * WA[ifoc]
      ComputeGemm(batch_size_, hidden_size_x4, attention_size_, T{1.0},
                  attention.cbegin(), attention.cend(),  // At-1
                  attention_size_,
                  weights_attn_ifoc_.cbegin(), weights_attn_ifoc_.cend(),  // WA[ifoc]
                  hidden_size_x4, T{1.0},
                  step_out_IFOC, output_ifoc_.end(),  // input contains Xt*(W[ifoc]^T)
                  hidden_size_x4);
    
      // calculate Xt*(W[ifoc]^T) + Ht-1*R[ifoc]
      ComputeGemm(batch_size_, hidden_size_x4, hidden_size_, T{1.0},
                  previous_state, previous_state_end,  // Ht-1
                  hidden_size_,
                  recurrent_weights_ifoc_.cbegin(), recurrent_weights_ifoc_.cend(),  // R[ifoc]
                  hidden_size_x4, T{1.0},
                  step_out_IFOC, output_ifoc_.end(),  // input contains Xt*(W[ifoc]^T)
                  hidden_size_x4);
    #else
      auto hidden_gemm_compute = [&](int thread_id) {
        int local_cols = hidden_size_x4 / hidden_num_threads_;
        int start_col = thread_id * local_cols;
        int compute_cols = local_cols;
        if (thread_id == hidden_num_threads_ - 1) {
          compute_cols = hidden_size_x4 - thread_id * local_cols;
        }

        // Xt*(W[ifoc]^T) = INPUTt * W[ifoc]^T + At-1 * WA[ifoc]^T
        ComputeGemm(batch_size_, compute_cols, attention_size_, T{1.0},
          attention.cbegin(), attention.cend(),  // At-1
          attention_size_,
                    weights_attn_ifoc_.cbegin() + start_col, weights_attn_ifoc_.cend(),  // WA[ifoc]
          hidden_size_x4, T{1.0},
                    step_out_IFOC + start_col, output_ifoc_.end(),  // input contains Xt*(W[ifoc]^T)
          hidden_size_x4);

        // calculate Xt*(W[ifoc]^T) + Ht-t*R[ifoc]
        ComputeGemm(batch_size_, compute_cols, hidden_size_, T{1.0},
                    previous_state, previous_state_end,  // Ht-1
                    hidden_size_,
                    recurrent_weights_ifoc_.cbegin() + start_col, recurrent_weights_ifoc_.cend(),  // R[ifoc]
                    hidden_size_x4, T{1.0},
                    step_out_IFOC + start_col, output_ifoc_.end(),  // input contains Xt*(W[ifoc]^T) + + At-1 * WA[ifoc]^T
                    hidden_size_x4);
      };

      ExecuteLambdaInParallel("Calculating Xt*(W[ifoc]^T) + Ht-1*R[ifoc])" + seqno_str,
                              hidden_gemm_compute, hidden_num_threads_, 1, ttp_, logger_);

    #endif

      span_T_iter batched_output, batched_output_end;
      if (output_sequence) {
        batched_output = outputs.begin() + step * output_step_length;
        batched_output_end = outputs.end();
      } else {
        batched_output = final_hidden_state.begin();
        batched_output_end = final_hidden_state.end();
      }

      span_T_iter step_out_IFOC_end = step_out_IFOC + batch_size_ * hidden_size_x4;
      GateComputations(step_out_IFOC, step_out_IFOC_end,
                       c_prev, C_prev_end,
                       c_prev_clipped, C_prev_clipped_end,
                       batched_output, batched_output_end,
                       sequence_lengths, min_sequence_length, step, 0, batch_size_, output_sequence);

      // copy last row to final_cell_state
      for (int lrow = 0; lrow < batch_size_; lrow++) {
        if ((step + 1) == sequence_lengths[lrow]) {
          auto src = batched_internal_memory_prev_.cbegin() + lrow * hidden_size_;
          auto dst = final_cell_state.begin() + lrow * hidden_size_;
          std::copy(src, src + hidden_size_, dst);
        }
      }

      if (output_sequence) {
        //set to 0 if step >= sequence_length
        for (int lrow = 0; lrow < batch_size_; lrow++) {
          if (step >= min_sequence_length && step >= sequence_lengths[lrow]) {
            auto dst = outputs.begin() + step * output_step_length + lrow * hidden_size_;
            std::fill(dst, dst + hidden_size_, T{});
          }
        }
      }

      previous_state = batched_output;
      previous_state_end = batched_output_end;

      attention_wrapper_.ProcessOutput(outputs.subspan(step * output_step_length, batch_size_ * hidden_size_));
    }
  }

  if (output_sequence) {
    // copy last output to final_hidden_state
    for (int i = 0; i < batch_size_; i++) {
      const int seq_len = sequence_lengths[i];
      span_T_const_iter src = outputs.cbegin() + (seq_len - 1) * output_step_length + i * hidden_size_;
      span_T_iter dest = final_hidden_state.begin() + i * hidden_size_;
      std::copy(src, src + hidden_size_, dest);
    }

    if (direction_ == Direction::kReverse)
      ReverseSequence<T>(outputs, original_outputs, sequence_lengths, max_sequence_length,
                         batch_size_, hidden_size_, num_directions);
  }
}


template <typename T>
void UniDirectionalAttnLstm<T>::GateComputations(span_T_iter& out, span_T_iter& out_end,
                                                 span_T_iter& C_prev, span_T_iter& C_prev_end,  // Ct-1 value not 'ct'. using 'C' for clarity
                                                 span_T_iter& C_prev_clipped, span_T_iter& C_prev_clipped_end,
                                                 span_T_iter& batched_output, span_T_iter& batched_output_end,
                                                 const gsl::span<const int>& seq_lengths,
                                                 const int min_sequence_length,
                                                 const int step,
                                                 const int row,
                                                 const int local_fused_hidden_rows,
                                                 bool output_sequence) {
  int hidden_size_x4 = 4 * hidden_size_;

  // Activation gates.
  for (int b = 0; b < local_fused_hidden_rows; b++) {
    if (step >= min_sequence_length && step >= seq_lengths[row + b]) {
      if (output_sequence) {
        auto fill_output = batched_output + (row + b) * hidden_size_;
        std::fill(fill_output, fill_output + hidden_size_, T{});
      }

      continue;
    }

    std::string row_str = " row[" + std::to_string(row + b) + "]";

    // check that we have hidden_size_x4 left starting at cur_out + b * hidden_size_x4, and get a raw pointer to that
    float* pi = SafeRawPointer<T>(out + b * hidden_size_x4, out_end, hidden_size_x4);
    float* pf = pi + hidden_size_;
    float* po = pf + hidden_size_;
    float* pc = po + hidden_size_;

    float* pCprev_hidden_size = SafeRawPointer<T>(C_prev + b * hidden_size_, C_prev_end, hidden_size_);

    // Input Gate
    if (use_peepholes_) {
      deepcpu::elementwise_product(pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_i_, 0, hidden_size_),
                                   pi, hidden_size_);
    }

    const float* pBi = use_bias_ ? SafeRawConstPointer<T>(bias_WRi_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBi, pi, hidden_size_);  // post: pi has input to f() to calculate i
    activation_f_.func(pi, hidden_size_, activation_f_.alpha, activation_f_.beta);
    //DumpMatrix("i" + row_str, pi, 1, hidden_size_);

    // Forget Gate
    if (input_forget_) {
      for (int i = 0; i < hidden_size_; i++) {
        pf[i] = 1.0f - pi[i];
      }
    } else {
      if (use_peepholes_) {
        deepcpu::elementwise_product(
          pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_f_, 0, hidden_size_), pf, hidden_size_);
      }

      const float* pBf = use_bias_ ? SafeRawConstPointer<T>(bias_WRf_, 0, hidden_size_) : nullptr;
      clip_with_bias_ptr_(clip_, pBf, pf, hidden_size_);
      activation_f_.func(pf, hidden_size_, activation_f_.alpha, activation_f_.beta);
    }

    // Block Gate
    const float* pBc = use_bias_ ? SafeRawConstPointer<T>(bias_WRc_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBc, pc, hidden_size_);
    activation_g_.func(pc, hidden_size_, activation_g_.alpha, activation_g_.beta);

    // C_current. use previous C value as input, and update in-place
    float* pC_cur = pCprev_hidden_size;
    deepcpu::merge_lstm_gates_to_memory(pCprev_hidden_size, pi, pf, pc, pC_cur, hidden_size_);

    // Output Gate
    if (use_peepholes_) {
      deepcpu::elementwise_product(
        pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_o_, 0, hidden_size_), po, hidden_size_);

    }

    // calculate 'ot'
    const float* pBo = use_bias_ ? SafeRawConstPointer<T>(bias_WRo_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBo, po, hidden_size_);
    activation_f_.func(po, hidden_size_, activation_f_.alpha, activation_f_.beta);
    // DumpMatrix("o" + row_str, po, 1, hidden_size_);

    // calculate 'Ht'
    float* pH = SafeRawPointer<T>(batched_output + row * hidden_size_ + b * hidden_size_,
                                  batched_output_end, hidden_size_);

    // the C_prev_clipped location is not actually used as input - it's temporary storage for writing
    // the clipped Ct value to, before calling h(). As such a) it could just be a local variable
    // of std::vector<float> with size of hidden_size_, b) the LotusRT version wasn't 'broken' by never
    // incrementing what C_prev_clipped pointed to.
    float* pC_prev_clipped = SafeRawPointer<T>(C_prev_clipped + b * hidden_size_, C_prev_clipped_end, hidden_size_);

    activation_h_.func(pC_cur, pC_prev_clipped, po, pH, hidden_size_, activation_h_.alpha, activation_h_.beta);
  }

  auto num_rows = local_fused_hidden_rows - row;
  std::string rows_str = " rows[" + std::to_string(row) + ".." + std::to_string(num_rows) + "]";

  DumpMatrix("i" + rows_str, &*out, num_rows, hidden_size_, 0, hidden_size_x4);
  DumpMatrix("f" + rows_str, &*out, num_rows, hidden_size_, 1 * hidden_size_, hidden_size_x4);
  DumpMatrix("o" + rows_str, &*out, num_rows, hidden_size_, 2 * hidden_size_, hidden_size_x4);
  DumpMatrix("c" + rows_str, &*out, num_rows, hidden_size_, 3 * hidden_size_, hidden_size_x4);
  DumpMatrix("C" + rows_str, &*C_prev, num_rows, hidden_size_);  // Ct overwrites the input C_prev value
  DumpMatrix("H" + rows_str, &*batched_output, num_rows, hidden_size_);
}

//The thread numbers are set based on profiling runs on Surface book,
//An old Xeon with 4 cores, and a relatively new xeon with 24 cores
template <typename T>
void UniDirectionalAttnLstm<T>::SetNumThreads() {
  int threads = std::thread::hardware_concurrency() - 1;

  if (threads < 1)
    threads = 1;

  int imt = threads;
  if (imt > 16 && hidden_size_ <= 256)
    imt = 16;

  if (imt > 24)
    imt = 24;

  // total number of operations in the call to ComputeGemm to apply the weights to the inputs
  auto work = seq_length_ * batch_size_ * hidden_size_ * 4 * input_size_;
  const double kMinTaskSize = 50000;  // this value is copied from Eigen. simplistic and could be refined.

  input_num_threads_ = std::max<int>(1, std::min<int>(imt, static_cast<int>(work / kMinTaskSize)));

  VLOGS(logger_, 1) << "Input Threads : " << input_num_threads_;

  int hmt = threads;
  batch_parallel_ = false;


  // For readability of the below logic

  // TODO: Temperately removed path: parallelize by partitioning the batch rows,
  //       and its logic in the Compute() method. Will evaluate and finialize it
  //       in later performance tuning stage.
  const auto num_columns = hidden_size_;
  {
    if (hmt > 2 && num_columns <= 128)
      hmt = 2;
    if (hmt > 5 && num_columns <= 256)
      hmt = 5;
    if (hmt > 7 && num_columns <= 512)
      hmt = 7;
    if (hmt > 11 && num_columns <= 1024)
      hmt = 11;

    hidden_num_threads_ = hmt;
  }

  VLOGS(logger_, 1) << "Hidden Threads : " << hidden_num_threads_;
}


template class UniDirectionalAttnLstm<float>;

}  // namespace detail
}  // namespace rnn
}  // namespace ml
}  // namespace onnxruntime
