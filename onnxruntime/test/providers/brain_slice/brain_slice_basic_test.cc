// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/providers/brainslice/fpga_handle.h"
#include "gtest/gtest.h"
#include "3rdparty/half.hpp"
#include "loopback_client.h"
#include "Lstm_client.h"

namespace onnxruntime {
namespace test {

TEST(BrainSliceBasicTest, MvMulTest) {
  fpga::FPGAInfo info = {0, false, "", "", ""};
  fpga::FPGAHandle handle(info);

  const BS_Capabilities& capacity = handle.GetCapacities();
  //1. prepare an 400 * 200 matrix
  int row = 400, col = 200;
  typedef half_float::half float16type;
  std::vector<float16type> half_m(row * col, float16type(1.0f));

  auto status = handle.LoadMatrix(half_m, row, col, 0, true, ISA_Mem_MatrixRf);
  EXPECT_TRUE(status.IsOK());

  std::vector<float16type> half_x(col, float16type(1.0f));

  BS_MVMultiplyParams param;
  param.numCols = col;
  param.numRows = row;
  param.startMaddr = 0;
  param.useDram = false;

  std::vector<float> result;

  status = handle.SendSync(
      [&](void* buffer, size_t* size) {
        return BS_CommonFunctions_MatrixVectorMultiply_Request_Float16(&capacity.m_bsParameters, &param, &half_x[0], buffer, size);
      },
      [&](void* buffer, size_t size) {
        const void* output;
        size_t n_output;
        auto status = BS_CommonFunctions_MatrixVectorMultiply_Response_Float16(&capacity.m_bsParameters, &param, buffer, size, &output, &n_output);
        if (status)
          return status;
        const float16type* out = static_cast<const float16type*>(output);
        for (size_t i = 0; i < n_output; i++)
          result.push_back(float(*(out + 1)));
        return status;
      });
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(result.size(), 400);
  for (auto f : result)
    EXPECT_EQ(f, 200.f);
}

TEST(BrainSliceBasicTest, DISABLED_LoopBackTest) {
  fpga::FPGAInfo info = {0, true, "testdata/loopback/instructions.bin", "testdata/loopback/data.bin", "testdata/loopback/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BS_Capabilities& capacity = handle.GetCapacities();

  using float16type = half_float::half;
  size_t native_dim = capacity.m_bsParameters.HWVEC_ELEMS;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float16type> half_x;
  size_t expect_dim = (x.size() + native_dim - 1) / native_dim;
  for (auto f : x)
    half_x.push_back(float16type(f));
  for (size_t i = half_x.size(); i < (expect_dim * native_dim); i++)
    half_x.push_back(float16type());
  void* input_ptr = &half_x[0];

  Example_Param param = {true, 6};
  Example_Result result;
  size_t result_size;
  std::vector<float> outputs;
  auto status = handle.SendSync(
      [&](void* buffer, size_t* size) { return Example_Model_Loopback_Request_Float16(&capacity.m_bsParameters, &param, input_ptr, buffer, size); },
      [&](void* buffer, size_t size) {
        const void* output;
        size_t count = 0;
        auto status = Example_Model_Loopback_Response_Float16(&capacity.m_bsParameters, &param, buffer, size, &result, &result_size, &output, &count);
        if (status)
          return status;
        const float16type* val = static_cast<const float16type*>(output);
        for (size_t i = 0; i < count; i++) {
          outputs.push_back(float(val[i]));
        }
        return status;
      });
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(outputs.size(), x.size());
  for (size_t i = 0; i < outputs.size(); i++)
    EXPECT_EQ(outputs[i], x[i]);
}

using float16 = half_float::half;
void Convert2Float16AndPad(const std::vector<std::vector<float>>& src,
                           std::vector<std::vector<float16>>& dst,
                           size_t num_rows, size_t num_cols) {
  for (size_t i = 0; i < src.size(); i++) {
    for (size_t j = 0; j < src[i].size(); j++) {
      dst[i].push_back(float16(src[i][j]));
    }
    dst[i].insert(dst[i].end(), num_cols - src[i].size(), float16());
  }

  std::vector<float16> zero(num_cols, float16());
  for (size_t k = src.size(); k < num_rows; k++) {
    dst.insert(dst.end(), num_rows - src.size(), zero);
  }
}

void Convert2Float16AndPad(const std::vector<std::vector<float>>& src,
                           std::vector<float16>& dst,
                           size_t num_rows, size_t num_cols) {
  for (size_t i = 0; i < src.size(); i++) {
    for (size_t j = 0; j < src[i].size(); j++) {
      dst.push_back(float16(src[i][j]));
    }
    for (size_t k = src[i].size(); k < num_cols; k++) {
      dst.push_back(float16());
    }
  }

  dst.insert(dst.end(), (num_rows - src.size()) * num_cols, float16());
}

void Pad(const std::vector<std::vector<float>>& src,
         std::vector<float>& dst,
         size_t num_rows, size_t num_cols) {
  for (size_t i = 0; i < src.size(); i++) {
    for (size_t j = 0; j < src[i].size(); j++) {
      dst.push_back(src[i][j]);
    }
    for (size_t k = src[i].size(); k < num_cols; k++) {
      dst.push_back(0.0f);
    }
  }

  dst.insert(dst.end(), (num_rows - src.size()) * num_cols, float16());
}

TEST(BrainSliceBasicTest, LotusRT_TestGRUOpForwardBasic) {
  fpga::FPGAInfo info = {0, false, "", "", ""};
  fpga::FPGAHandle handle(info);

  const BS_Capabilities& capabilities = handle.GetCapacities();

  uint32_t native_dim = capabilities.m_bsParameters.HWVEC_ELEMS;
  uint32_t input_dim = 100;
  uint32_t output_dim = 100;
  const int input_tiles = input_dim / native_dim;
  const int output_tiles = output_dim / native_dim;

  std::vector<std::vector<float>> bias = {
      {0.381619f, 0.0323954f},   // Wbz
      {-0.258721f, 0.45056f},    // Wbr
      {-0.250755f, 0.0967895f},  // Wbh
  };
  std::vector<std::vector<float16>> half_bias(bias.size());
  Convert2Float16AndPad(bias, half_bias, bias.size(), output_dim);

  // Initialize GRU and Load Bias
  Lstm_InitGruParams init_args;
  init_args.inputDim = input_dim;
  init_args.outputDim = output_dim;
  auto status = handle.SendSync(
      [&](void* request, size_t* request_size) {
        void* zero = _alloca(init_args.outputDim * sizeof(float16));
        memset(zero, 0, init_args.outputDim * sizeof(float16));
        const void* bias_addr[3];
        std::transform(half_bias.begin(), half_bias.end(), bias_addr, [](auto& v) { return v.data(); });

        return Lstm_Functions_InitGruAndLoadBias_Request_Float16(
            &capabilities.m_bsParameters,
            &init_args,
            zero,
            bias_addr,
            request, request_size);
      },
      [&](const void* response, size_t response_size) {
        return Lstm_Functions_InitGruAndLoadBias_Response(
            &capabilities.m_bsParameters,
            response, response_size);
      });

  // Load weights matrix
  uint32_t matAddr = 0;
  std::vector<std::vector<float>> Wr = {{-0.0091708f, -0.255364f}, {-0.106952f, -0.266717f}};
  std::vector<float16> half_Wr;
  Convert2Float16AndPad(Wr, half_Wr, output_dim, input_dim);
  status = handle.LoadMatrix(half_Wr, native_dim, native_dim, matAddr, true, ISA_Mem_MatrixRf);
  matAddr++;

  std::vector<std::vector<float>> Rr = {{-0.228172f, 0.405972f}, {0.31576f, 0.281487f}};
  std::vector<float16> half_Rr;
  Convert2Float16AndPad(Rr, half_Rr, output_dim, output_dim);
  status = handle.LoadMatrix(half_Rr, native_dim, native_dim, matAddr, true, ISA_Mem_MatrixRf);
  matAddr++;

  std::vector<std::vector<float>> Wz = {{-0.494659f, 0.0453352f}, {-0.487793f, 0.417264f}};
  std::vector<float16> half_Wz;
  Convert2Float16AndPad(Wz, half_Wz, output_dim, input_dim);
  status = handle.LoadMatrix(half_Wz, native_dim, native_dim, matAddr, true, ISA_Mem_MatrixRf);
  matAddr++;

  std::vector<std::vector<float>> Rz = {{0.146626f, -0.0620289f}, {-0.0815302f, 0.100482f}};
  std::vector<float16> half_Rz;
  Convert2Float16AndPad(Rz, half_Rz, output_dim, output_dim);
  status = handle.LoadMatrix(half_Rz, native_dim, native_dim, matAddr, true, ISA_Mem_MatrixRf);
  matAddr++;

  std::vector<std::vector<float>> Wh = {{-0.0888852f, -0.428709f}, {-0.283349f, 0.208792f}};
  std::vector<float16> half_Wh;
  Convert2Float16AndPad(Wh, half_Wh, output_dim, input_dim);
  status = handle.LoadMatrix(half_Wh, native_dim, native_dim, matAddr, true, ISA_Mem_MatrixRf);
  matAddr++;

  std::vector<std::vector<float>> Rh = {{-0.394864f, 0.42111f}, {-0.386624f, -0.390225f}};
  std::vector<float16> half_Rh;
  Convert2Float16AndPad(Rh, half_Rh, output_dim, output_dim);
  status = handle.LoadMatrix(half_Rh, native_dim, native_dim, matAddr, true, ISA_Mem_MatrixRf);
  matAddr++;

  // Evaluate GRU
  std::vector<std::vector<float>> X = {{-0.455351f, -0.276391f},
                                       {-0.185934f, -0.269585f}};
  std::vector<std::vector<float16>> half_X(X.size());
  Convert2Float16AndPad(X, half_X, X.size(), input_dim);

  Lstm_EvalGruParams eval_args;
  eval_args.rnnSteps = (uint32_t)half_X.size();
  eval_args.inputDim = input_dim;
  eval_args.outputDim = output_dim;
  eval_args.exportHidden = 1;  // Do not export hidden state in TP1

  std::vector<float> initial_h = {0.0f, 0.0f};
  std::vector<std::vector<float>> expected_Y = {{-0.03255286f, 0.0774838f},
                                                {-0.05556786f, 0.0785508f}};
  std::vector<float> expected_Y_Pad;
  Pad(expected_Y, expected_Y_Pad, eval_args.rnnSteps, output_dim);

  std::vector<float> expected_Y_h = {-0.05556786f, 0.0785508f};
  std::vector<float16> half_Y;
  std::vector<float> Y;

  status = handle.SendSync(
      [&](void* request, size_t* request_size) {
        auto addr_X = static_cast<const void**>(_alloca(eval_args.rnnSteps * sizeof(const void*)));
        std::transform(half_X.begin(), half_X.end(), addr_X, [](auto& v) { return v.data(); });

        auto status = Lstm_Functions_EvaluateGru_Request_Float16(
            &capabilities.m_bsParameters,
            &eval_args,
            addr_X,
            request, request_size);

        return status;
      },
      [&](const void* response, size_t response_size) {
        size_t output_size, output_count = (eval_args.exportHidden ? eval_args.rnnSteps : 1);
        auto addr_Y = static_cast<const void**>(_alloca(output_count * sizeof(const void*)));

        auto status = Lstm_Functions_EvaluateGru_Response_Float16(
            &capabilities.m_bsParameters,
            &eval_args,
            response, response_size,
            addr_Y, &output_size, &output_count);

        half_Y.resize(output_size * output_count);

        for (auto i = 0; i < output_count; ++i)
          memcpy(&half_Y[i * output_size], addr_Y[i], output_size * sizeof(float16));

        for (auto v : half_Y) {
          Y.push_back(v);
        }

        return status;
      });

  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(Y.size(), expected_Y_Pad.size());

  for (size_t i = 0; i < expected_Y_Pad.size(); i++) {
    EXPECT_NEAR(Y[i], expected_Y_Pad[i], 1e-1);
  }
}

}  // namespace test
}  // namespace onnxruntime
