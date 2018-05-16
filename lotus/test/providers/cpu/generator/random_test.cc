#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <algorithm>
#include <random>

namespace Lotus {
namespace Test {

TEST(Random, RandomNormal2DDouble) {
  OpTester test("RandomNormal");

  std::vector<int64_t> dims{20, 50};

  float scale = 10.f;
  float mean = 0.f;
  float seed = 123.f;

  test.AddAttribute("scale", scale);
  test.AddAttribute("mean", mean);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", TensorProto::DOUBLE);
  test.AddAttribute("shape", dims);

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<double> distribution{mean, scale};

  std::vector<double> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](double &value) { value = distribution(generator); });

  test.AddOutput<double>("Y", dims, expected_output);
  test.Run();
}

void RunRandomNormalLike3DFloat(bool infer_dtype = false) {
  OpTester test("RandomNormalLike");

  std::vector<int64_t> dims{2, 2, 3};

  float scale = 10.f;
  float mean = 0.f;
  float seed = 123.f;

  test.AddAttribute("scale", scale);
  test.AddAttribute("mean", mean);
  test.AddAttribute("seed", seed);

  if (!infer_dtype)
    test.AddAttribute<int64_t>("dtype", TensorProto::FLOAT);

  test.AddInput<float>("X", dims,
                       {0.f, 0.f, 0.f,
                        0.f, 0.f, 0.f,

                        0.f, 0.f, 0.f,
                        0.f, 0.f, 0.f});

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  std::vector<float> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](float &value) { value = distribution(generator); });

  test.AddOutput<float>("Y", dims, expected_output);

  test.Run();
}

TEST(Random, RandomNormalLike3DDouble) {
  RunRandomNormalLike3DFloat();
}

TEST(Random, RandomNormalLikeInferDType) {
  const bool infer_dtype = true;
  RunRandomNormalLike3DFloat(infer_dtype);
}

TEST(Random, RandomUniform1DFloat) {
  OpTester test("RandomUniform");

  std::vector<int64_t> dims{10};

  float low = 0.f;
  float high = 100.f;
  float seed = 123.f;

  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", TensorProto::FLOAT);
  test.AddAttribute("shape", dims);

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::uniform_real_distribution<float> distribution{low, high};

  std::vector<float> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](float &value) { value = distribution(generator); });

  test.AddOutput<float>("Y", dims, expected_output);

  test.Run();
}

void RunRandomUniformLikeTest(bool infer_dtype = false) {
  OpTester test("RandomUniformLike");

  std::vector<int64_t> dims{2, 6};

  float low = 0.f;
  float high = 100.f;
  float seed = 123.f;

  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);

  if (!infer_dtype)
    test.AddAttribute<int64_t>("dtype", TensorProto::DOUBLE);

  test.AddInput<double>("X", dims,
                        {0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0.});

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::uniform_real_distribution<double> distribution{low, high};

  std::vector<double> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](double &value) { value = distribution(generator); });

  test.AddOutput<double>("Y", dims, expected_output);

  test.Run();
}

TEST(Random, RandomUniformLike2DDouble) {
  RunRandomUniformLikeTest();
}

TEST(Random, RandomUniformLikeInferDType) {
  const bool infer_dtype = true;
  RunRandomUniformLikeTest(infer_dtype);
}

TEST(Random, InvalidDType) {
  float seed = 123.f;

  std::vector<int64_t> dims{1, 4};
  std::vector<int32_t> input{0, 0, 0, 0};
  std::vector<double> expected_output{0., 0., 0., 0.};

  {
    OpTester test("RandomNormal");

    float scale = 10.f;
    float mean = 0.f;

    test.AddAttribute("scale", scale);
    test.AddAttribute("mean", mean);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);
    test.AddAttribute("shape", dims);

    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid dtype");
  }

  {
    OpTester test("RandomUniform");

    float low = 0.f;
    float high = 100.f;

    test.AddAttribute("low", low);
    test.AddAttribute("high", high);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);
    test.AddAttribute("shape", dims);

    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid dtype");
  }

  {
    OpTester test("RandomNormalLike");

    float scale = 10.f;
    float mean = 0.f;

    test.AddAttribute("scale", scale);
    test.AddAttribute("mean", mean);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);

    test.AddInput<int32_t>("X", dims, input);
    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid dtype");
  }

  {
    OpTester test("RandomUniformLike");

    float low = 0.f;
    float high = 100.f;

    test.AddAttribute("low", low);
    test.AddAttribute("high", high);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);

    test.AddInput<int32_t>("X", dims, input);
    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid dtype");
  }
}

}  // namespace Test
}  // namespace Lotus
