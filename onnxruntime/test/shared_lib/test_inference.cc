// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#include <memory>
#include <vector>
#include <iostream>
#include <atomic>
#include <gtest/gtest.h>
#include "test_allocator.h"

using namespace onnxruntime;

void RunSession(ONNXRuntimeAllocator* env, ONNXSession* session_object,
                const std::vector<size_t>& dims_x,
                const std::vector<float>& values_x,
                const std::vector<size_t>& dims_y,
                const std::vector<float>& values_y) {
  std::unique_ptr<ONNXValue, decltype(&ReleaseONNXValue)> value_x(nullptr, ReleaseONNXValue);
  std::vector<ONNXValuePtr> inputs(1);
  inputs[0] = ONNXRuntimeCreateTensorAsONNXValue(env, dims_x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  value_x.reset(inputs[0]);
  void* raw_data;
  ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetTensorMutableData(inputs[0], &raw_data));
  memcpy(raw_data, values_x.data(), values_x.size() * sizeof(values_x[0]));
  std::vector<const char*> input_names{"X"};
  std::unique_ptr<ONNXValueList, decltype(&ReleaseONNXValueList)> output(nullptr, ReleaseONNXValueList);
  size_t output_len;
  {
    ONNXValueListPtr t;
    ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeRunInferenceAndFetchAll(session_object, input_names.data(), inputs.data(), inputs.size(), &t, &output_len));
    output.reset(t);
  }

  ASSERT_EQ(static_cast<size_t>(1), output_len);
  ONNXValuePtr rtensor = ONNXRuntimeONNXValueListGetNthValue(output.get(), 0);
  size_t rtensor_dims;
  ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetTensorShapeDimCount(rtensor, &rtensor_dims));
  std::vector<size_t> shape_array(rtensor_dims);
  ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetTensorShape(rtensor, shape_array.data(), shape_array.size()));
  ASSERT_EQ(shape_array, dims_y);
  size_t total_len = 1;
  for (size_t i = 0; i != rtensor_dims; ++i) {
    total_len *= shape_array[i];
  }
  ASSERT_EQ(values_y.size(), total_len);
  float* f;
  ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetTensorMutableData(rtensor, (void**)&f));
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}


template <typename T>
void TestInference(ONNXEnv* env, const T& model_uri,
                   const std::vector<size_t>& dims_x,
                   const std::vector<float>& values_x,
                   const std::vector<size_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   bool enable_cuda, bool custom_op) {
  SessionOptionsWrapper sf(env);

  if (enable_cuda) {
#ifdef USE_CUDA
    ONNXRuntimeProviderFactoryPtr* f;
    ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeCreateCUDAExecutionProviderFactory(0, &f));
    sf.AppendExecutionProvider(f);
    ONNXRuntimeReleaseObject(f);
#else
    FAIL() << "CUDA is not enabled";
#endif
  }
  if (custom_op) {
    sf.AddCustomOp("liblotus_custom_op_shared_lib_test.so");
  }
  std::unique_ptr<ONNXSession, decltype(&ReleaseONNXSession)> inference_session(sf.ONNXRuntimeCreateInferenceSession(model_uri.c_str()), ReleaseONNXSession);
  MockedONNXRuntimeAllocator alloca;
  // Now run
  RunSession((ONNXRuntimeAllocator*)&alloca, inference_session.get(), dims_x, values_x, expected_dims_y, expected_values_y);
  alloca.LeakCheck();
}

void ONNXRUNTIME_API_STATUSCALL MyLoggingFunction(void*, ONNXRuntimeLoggingLevel, const char*, const char*, const char*, const char*) {
}

template <bool use_customer_logger>
class CApiTestImpl : public ::testing::Test {
 protected:
  ONNXEnv* env;

  void SetUp() override {
    if (use_customer_logger) {
      ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeInitializeWithCustomLogger(MyLoggingFunction, nullptr, ONNXRUNTIME_LOGGING_LEVEL_kINFO, "Default", &env));
    } else {
      ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeInitialize(ONNXRUNTIME_LOGGING_LEVEL_kINFO, "Default", &env));
    }
  }

  void TearDown() override {
    ReleaseONNXEnv(env);
  }

  // Objects declared here can be used by all tests in the test case for Foo.
};

typedef CApiTestImpl<false> CApiTest;

#ifdef _WIN32
typedef std::wstring PATH_TYPE;
#define TSTR(X) L##X
#else
#define TSTR(X) (X)
typedef std::string PATH_TYPE;
#endif

static const PATH_TYPE MODEL_URI = TSTR("testdata/mul_1.pb");
static const PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_1.pb");

// Tests that the Foo::Bar() method does Abc.
TEST_F(CApiTest, simple) {
  const PATH_TYPE input_filepath = TSTR("this/package/testdata/myinputfile.dat");
  const PATH_TYPE output_filepath = TSTR("this/package/testdata/myoutputfile.dat");
  // simple inference test
  // prepare inputs
  std::cout << "Running simple inference" << std::endl;
  std::vector<size_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<size_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference(env, MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, false, false);
#if USE_CUDA
  TestInference(env, MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, true, false);
#endif
}

#ifndef _WIN32
//doesn't work, failed in type comparison
TEST_F(CApiTest, DISABLED_custom_op) {
  std::cout << "Running custom op inference" << std::endl;
  std::vector<size_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<size_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  TestInference(env, CUSTOM_OP_MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, false, true);
}
#endif

TEST_F(CApiTest, create_tensor) {
  const char* s[] = {"abc", "kmp"};
  size_t expected_len = 2;
  MockedONNXRuntimeAllocator alloca;
  {
    std::unique_ptr<ONNXValue, decltype(&ReleaseONNXValue)> tensor(
        ONNXRuntimeCreateTensorAsONNXValue((ONNXRuntimeAllocator*)&alloca, {expected_len}, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING), ReleaseONNXValue);
    ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeFillStringTensor(tensor.get(), s, expected_len));
    size_t len;
    ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetTensorShapeElementCount(tensor.get(), &len));
    ASSERT_EQ(len, expected_len);
    size_t datalen;
    ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetStringTensorDataLength(tensor.get(), &datalen));
    std::string result(datalen, '\0');
    std::vector<size_t> offsets(len);
    ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetStringTensorContent(tensor.get(), (void*)result.data(), datalen, offsets.data(), offsets.size()));
  }
  alloca.LeakCheck();
}

TEST_F(CApiTest, create_tensor_with_data) {
  float values[] = {3.0, 1.0, 2, 0};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
  ONNXRuntimeAllocatorInfo* info;
  ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeCreateAllocatorInfo("Cpu", ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeDefault, &info));
  std::vector<size_t> dims = {3};
  std::unique_ptr<ONNXValue, decltype(&ReleaseONNXValue)> tensor(
      ONNXRuntimeCreateTensorWithDataAsONNXValue(info, values, values_length * sizeof(float), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), ReleaseONNXValue);
  ReleaseONNXRuntimeAllocatorInfo(info);
  void* new_pointer;
  ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeGetTensorMutableData(tensor.get(), &new_pointer));
  ASSERT_EQ(new_pointer, values);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
