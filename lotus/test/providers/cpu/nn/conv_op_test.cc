#include "core/graph/utils.h"
#include "core/providers/cpu/nn/conv.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
namespace Test {

static const TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};

TEST(ConvTest, Conv) {
  LotusIR::NodeArg input1_def("X", &s_typeProto_float), input2_def("W", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Conv", {&input1_def, &input2_def}, {&output_def});
  model.Node().AddAttribute("auto_pad", "");
  model.Node().AddAttribute("strides", vector<int64_t>{1});
  model.Node().AddAttribute("dilations", vector<int64_t>{1});
  model.Node().AddAttribute("pads", vector<int64_t>{0, 0});
  model.Node().AddAttribute("kernel_shape", vector<int64_t>{1});
  model.Node().AddAttribute("group", int64_t(1));
  SimpleFloatTest<Conv> test(model);

  std::vector<float> X = {-0.21559301018714905f, 0.4691687822341919f, 0.4426700472831726f, -0.4517466723918915f,
                          -0.05216419696807861f, 0.29067182540893555f, 0.251010000705719f};
  std::vector<int64_t> X_shape = {1, 1, 7};
  std::vector<float> W = {0.24472862482070923f};
  std::vector<int64_t> W_shape = {1, 1, 1};
  std::vector<int64_t> Y_shape = {1, 1, 7};
  test.AddInput(X_shape, X);
  test.AddInput(W_shape, W);
  test.AddOutput(Y_shape);
  float expected_vals[] = {-0.052761781960725784f, 0.11481902748346329f, 0.10833403468132019f, -0.11055534332990646f,
                           -0.012766072526574135f, 0.07113571465015411f, 0.061429332941770554f};
  test.Run(Y_shape, expected_vals);
}
}  // namespace Test
}  // namespace Lotus
