#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"
#include <exception>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/inference_session.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace Lotus::Logging;

namespace Lotus {
namespace Test {

// Check functions for tensor types

// The default implementation compares for equality, specialized versions for other types are below
template <typename T>
void Check(const OpTester::Data& expected_data, const Tensor& output_tensor) {
  auto& expected_tensor = expected_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.Data<T>();
  auto* output = output_tensor.Data<T>();
  auto size = output_tensor.Shape().Size();

  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

template <>
void Check<float>(const OpTester::Data& expected_data, const Tensor& output_tensor) {
  auto& expected_tensor = expected_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.Data<float>();
  auto* output = output_tensor.Data<float>();
  auto size = output_tensor.Shape().Size();

  bool has_abs_err = expected_data.absolute_error_.has_value();
  bool has_rel_err = expected_data.relative_error_.has_value();

  for (int i = 0; i < size; ++i) {
    if (std::isinf(expected[i]))  // Test infinity for equality
      EXPECT_EQ(expected[i], output[i]);
    else {
      if (!has_abs_err && !has_rel_err) {
        // the default for existing tests
        EXPECT_NEAR(expected[i], output[i], 0.001f);
      } else {
        if (has_abs_err) {
          EXPECT_NEAR(expected[i], output[i], expected_data.absolute_error_.value());
        }
        if (has_rel_err) {
          EXPECT_NEAR(expected[i], output[i], expected_data.relative_error_.value() * std::abs(expected[i]));
        }
      }
    }
  }
}

template <typename Type>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, const Tensor& output_tensor) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, output_tensor);
  else
    LOTUS_THROW("OpTester:Check() not implemented for output tensor type of ", type);
}

template <typename Type, typename Next, typename... Types>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, const Tensor& output_tensor) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, output_tensor);
  else
    CheckDispatch<Next, Types...>(type, expected_data, output_tensor);
}

void Check(const OpTester::Data& expected_data, const Tensor& output_tensor) {
  LOTUS_ENFORCE(expected_data.data_.Get<Tensor>().Shape() == output_tensor.Shape(),
                "Expected output shape [" + expected_data.data_.Get<Tensor>().Shape().ToString() +
                    "] did not match run output shape [" +
                    output_tensor.Shape().ToString() + "]");

  CheckDispatch<bool, float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, std::string, MLFloat16>(output_tensor.DataType(), expected_data, output_tensor);
}

// Check for non tensor types

template <typename T>
void Check(const OpTester::Data& expected_data, const T& run_output) {
  EXPECT_EQ(expected_data.data_.Get<T>(), run_output);
}

template <typename Type>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, MLValue& mlvalue) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, mlvalue.Get<Type>());
  else
    LOTUS_THROW("OpTester:Check() not implemented for output tensor type of ", type);
}

template <typename Type, typename Next, typename... Types>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, MLValue& mlvalue) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, mlvalue.Get<Type>());
  else
    CheckDispatch<Next, Types...>(type, expected_data, mlvalue);
}

void Check(const OpTester::Data& expected_data, MLValue& mlvalue) {
  CheckDispatch<VectorMapStringToFloat, VectorMapInt64ToFloat>(expected_data.data_.Type(), expected_data, mlvalue);
}

OpTester::~OpTester() {
#if _DEBUG
  if (!run_called_) {
    std::cerr << "Someone forgot to call OpTester::Run()" << std::endl;
    __debugbreak();
  }
#endif
}

void OpTester::FillFeedsAndOutputNames(const std::vector<LotusIR::NodeArg*>&,
                                       const std::vector<LotusIR::NodeArg*>& output_defs,
                                       std::unordered_map<std::string, MLValue>& feeds,
                                       std::vector<std::string>& output_names) {
  for (auto& output : output_defs) {
    if (output->Exists())
      output_names.push_back(output->Name());
  }

  for (auto& input : input_data_) {
    if (input.def_.Exists())
      feeds[input.def_.Name()] = input.data_;
  }
}

void OpTester::SetOutputAbsErr(const char* name, float v) {
  auto it = std::find_if(
      output_data_.begin(),
      output_data_.end(),
      [name](Data& data) {
        return (data.def_.Name() == name);
      });
  LOTUS_ENFORCE(it != output_data_.end());
  it->absolute_error_ = optional<float>(v);
}

void OpTester::SetOutputRelErr(const char* name, float v) {
  auto it = std::find_if(
      output_data_.begin(),
      output_data_.end(),
      [name](Data& data) {
        return (data.def_.Name() == name);
      });
  LOTUS_ENFORCE(it != output_data_.end());
  it->relative_error_ = optional<float>(v);
}

void OpTester::Run(ExpectResult expect_result, const std::string& expected_failure_string, LotusIR::ProviderType provider_type) {
  try {
#if _DEBUG
    run_called_ = true;
#endif
    // Generate the input & output def lists
    std::vector<LotusIR::NodeArg*> input_defs;
    std::vector<LotusIR::NodeArg*> output_defs;

    for (auto& data : input_data_) {
      input_defs.push_back(&data.def_);
    }

    for (auto& data : output_data_) {
      output_defs.push_back(&data.def_);
    }

    // Create a simple model
    auto p_model = std::make_unique<LotusIR::Model>("test");
    LotusIR::Graph* graph = p_model->MainGraph();
    auto& node = *graph->AddNode("node1", op_, op_, input_defs, output_defs, nullptr, domain_);

    // Add the attributes if any
    for (auto& add_attribute_fn : add_attribute_funcs_)
      add_attribute_fn(node);

    node.SetExecutionProviderType(provider_type);
    Status status = graph->Resolve();
    //LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());
    if (!status.IsOK()) {
      if (expect_result == ExpectResult::kExpectFailure) {
        EXPECT_TRUE(!status.IsOK());
        EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
      } else {
        LOGS_DEFAULT(ERROR) << "Resolve failed with status: " << status.ErrorMessage();
        EXPECT_TRUE(status.IsOK());
      }
    }
    if (!status.IsOK()) {
      return;
    }

    // Hookup the inputs and outputs
    std::unordered_map<std::string, MLValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

    // Run the model
    SessionOptions so;
    so.session_logid = op_;
    so.session_log_verbosity_level = 1;

    InferenceSession session_object{so};

    if (provider_type == LotusIR::kCudaExecutionProvider) {
#ifdef USE_CUDA
      CUDAExecutionProviderInfo epi;
      epi.device_id = 0;
      EXPECT_TRUE(session_object.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(epi)).IsOK());
#endif
    }

    status = session_object.Load(std::move(p_model));
    EXPECT_TRUE(status.IsOK());
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
      return;
    }

    status = session_object.Initialize();
    if (!status.IsOK()) {
      if (expect_result == ExpectResult::kExpectFailure) {
        EXPECT_TRUE(!status.IsOK());
        EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
      } else {
        LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
        EXPECT_TRUE(status.IsOK());
      }
    }
    if (!status.IsOK()) {
      return;
    }

    RunOptions run_options;
    run_options.run_tag = op_;
    run_options.run_log_verbosity_level = 1;
    std::vector<MLValue> fetches;
    status = session_object.Run(run_options, feeds, output_names, &fetches);
    if (status.IsOK()) {
      EXPECT_TRUE(expect_result == ExpectResult::kExpectSuccess);
      if (expect_result == ExpectResult::kExpectFailure) {
        return;
      }
    } else {
      if (expect_result == ExpectResult::kExpectFailure) {
        EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
      } else {
        LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
        EXPECT_TRUE(status.IsOK());
      }
      return;
    }

    // Verify the outputs
    // Todo: support check output with map/sequence/....
    size_t idx = 0;
    for (auto& expected_data : output_data_) {
      MLValue& mlvalue = fetches[idx];
      if (mlvalue.Fence())
        mlvalue.Fence()->BeforeUsingAsInput(LotusIR::kCpuExecutionProvider, 0);

      if (expected_data.def_.Exists()) {  // optional outputs won't exist
        if (expected_data.data_.IsTensor()) {
          Check(expected_data, mlvalue.Get<Tensor>());
        } else {
          Check(expected_data, mlvalue);
        }
        ++idx;

        // skip missing trailing optional outputs
        if (idx == fetches.size())
          break;
      }
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    // rethrow as some tests for error handling expect this
    throw;
  }
}

void OpTester::RunOnCpuAndCuda(ExpectResult expect_result, const std::string& expected_failure_string) {
  Run(expect_result, expected_failure_string, LotusIR::kCpuExecutionProvider);
#ifdef USE_CUDA
  Run(expect_result, expected_failure_string, LotusIR::kCudaExecutionProvider);
#endif
}

}  // namespace Test
}  // namespace Lotus
