// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/providers/brainslice/fpga_handle.h"
#include "gtest/gtest.h"
#include "3rdparty/half.hpp"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"

namespace onnxruntime {
namespace test {
static void VerifyOutputs(const std::vector<MLValue>& fetches,
                          const std::vector<int64_t>& expected_dims,
                          const std::vector<float>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.Data<float>(), rtensor.Data<float>() + expected_values.size());
  for (auto i = 0; i < found.size(); ++i)
    ASSERT_NEAR(expected_values[i], found[i], 1e-1);
}

//TODO: refactory this to avoid duplicate code
static void RunModel(InferenceSession& session_object,
                     const RunOptions& run_options,
                     bool is_preallocate_output_vec = false) {
  // prepare inputs
  std::vector<int64_t> dims_mul_x = {2, 1, 2};
  std::vector<float> values_mul_x = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};
  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(ONNXRuntimeMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  if (is_preallocate_output_vec) {
    fetches.resize(output_names.size());
    for (auto& elem : fetches) {
      CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(ONNXRuntimeMemTypeDefault), dims_mul_x, values_mul_x, &elem);
    }
  }

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {2, 1, 1, 2};
  std::vector<float> expected_values_mul_y = {-0.03255286f, 0.0774838f, -0.05556786f, 0.0785508f};

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_y, expected_values_mul_y);
}
static const std::string MODEL_URI = "testdata/gru_1.pb";

TEST(BrainSliceExecutionProviderTest, BasicTest) {
  fpga::FPGAInfo info = {0, false, "", "", ""};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  RunModel(session_object, run_options);
}
}  // namespace test
}  // namespace onnxruntime
