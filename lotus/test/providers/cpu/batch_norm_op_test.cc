#include "core/graph/utils.h"
#include "core/providers/cpu/nn/batch_norm.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/test_utils.h"
#include "core/framework/tensor.h"
#include "core/framework/inference_session.h"

using namespace std;

namespace Lotus {
namespace Test {

static const TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};
using InputDataMap = unordered_map<string, vector<float>>;
using InputShapesMap = unordered_map<string, vector<int64_t>>;

template <size_t count>
void TestBatchNorm(const string& test_name,
                   const InputDataMap& input_data_map,
                   const InputShapesMap& input_shapes_map,
                   float epsilon,
                   const float (&expected_output)[count],
                   const vector<int64_t>& expected_output_shape) {
  LotusIR::NodeArg
      input1_def("X", &s_typeProto_float),
      input2_def("scale", &s_typeProto_float),
      input3_def("B", &s_typeProto_float),
      input4_def("mean", &s_typeProto_float),
      input5_def("var", &s_typeProto_float),
      output_def("Y", &s_typeProto_float);

  vector<LotusIR::NodeArg*> input_defs{&input1_def, &input2_def, &input3_def, &input4_def, &input5_def};
  vector<LotusIR::NodeArg*> output_defs{&output_def};

  TestModel model(test_name.c_str(), input_defs, output_defs);
  model.Node().AddAttribute("epsilon", epsilon);

  SimpleFloatTest<BatchNorm> test(model);
  test.AddInput(input_shapes_map.at("X"), input_data_map.at("X"));
  test.AddInput(input_shapes_map.at("scale"), input_data_map.at("scale"));
  test.AddInput(input_shapes_map.at("B"), input_data_map.at("B"));
  test.AddInput(input_shapes_map.at("mean"), input_data_map.at("mean"));
  test.AddInput(input_shapes_map.at("var"), input_data_map.at("var"));
  test.AddOutput(expected_output_shape);
  test.Run(expected_output_shape, expected_output);
}

TEST(BatchNormTest, PositiveTestCase) {
  // This input was taken from the LotusRT SpatialBN_1.pb, SpatialBN_1_input.pb and SpatialBN_1_output.pb files.
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  float expected_output[] = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                             1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                             0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                             0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                             1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                             0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  TestBatchNorm("PositiveTest", input_data_map, input_shapes_map, epsilon, expected_output, expected_output_shape);
}

TEST(BatchNormTest, InvalidXDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7, 1}});  // invalid
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  float expected_output[] = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                             1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                             0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                             0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                             1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                             0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  try {
    TestBatchNorm("Invalid X Dim", input_data_map, input_shapes_map, epsilon, expected_output, expected_output_shape);
  } catch (const std::exception& ex) {
    EXPECT_THAT(ex.what(), testing::HasSubstr("Invalid input X"));
  }
}

TEST(BatchNormTest, InvalidScaleDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f, 0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1, 2}});  // invalid
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  float expected_output[] = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                             1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                             0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                             0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                             1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                             0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  try {
    TestBatchNorm("Invalid scale Dim", input_data_map, input_shapes_map, epsilon, expected_output, expected_output_shape);
  } catch (const std::exception& ex) {
    EXPECT_THAT(ex.what(), testing::HasSubstr("Invalid input scale"));
  }
}

TEST(BatchNormTest, InvalidBDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f, -0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1, 2}});  // invalid
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  float expected_output[] = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                             1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                             0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                             0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                             1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                             0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  try {
    TestBatchNorm("Invalid B Dim", input_data_map, input_shapes_map, epsilon, expected_output, expected_output_shape);
  } catch (const std::exception& ex) {
    EXPECT_THAT(ex.what(), testing::HasSubstr("Invalid input B"));
  }
}

TEST(BatchNormTest, InvalidMeanDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f, -2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1, 2}});  // invalid
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  float expected_output[] = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                             1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                             0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                             0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                             1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                             0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  try {
    TestBatchNorm("Invalid mean Dim", input_data_map, input_shapes_map, epsilon, expected_output, expected_output_shape);
  } catch (const std::exception& ex) {
    EXPECT_THAT(ex.what(), testing::HasSubstr("Invalid input mean"));
  }
}

TEST(BatchNormTest, InvalidVarDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f, 1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1, 2}});  // invalid

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  float expected_output[] = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                             1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                             0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                             0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                             1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                             0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  try {
    TestBatchNorm("Invalid var dim", input_data_map, input_shapes_map, epsilon, expected_output, expected_output_shape);
  } catch (const std::exception& ex) {
    EXPECT_THAT(ex.what(), testing::HasSubstr("Invalid input var"));
  }
}
}  // namespace Test
}  // namespace Lotus
