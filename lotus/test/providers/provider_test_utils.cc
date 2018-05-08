#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"

#include <exception>
#include <memory>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_common.h"
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

  CheckDispatch<bool, float, double, uint8_t, uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, std::string>(output_tensor.DataType(), expected_data, output_tensor);
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

OpTester::OpTester(const std::string& provider, const char* op, const char* domain)
    : provider_name_(provider), op_(op), domain_(domain) {
#ifdef USE_CUDA
  if (provider_name_ == LotusIR::kCudaExecutionProvider) {
    CUDAExecutionProviderInfo epi;
    epi.device_id = 0;
    provider_ = std::make_unique<CUDAExecutionProvider>(epi);
  }
#endif
}

OpTester::~OpTester() {
#if _DEBUG
  if (!run_called_) {
    std::cerr << "Someone forgot to call OpTester::Run()" << std::endl;
    __debugbreak();
  }
#endif
}

void OpTester::FillFeedsAndOutputNames(const std::vector<LotusIR::NodeArg*>& input_defs,
                                       const std::vector<LotusIR::NodeArg*>& output_defs,
                                       std::unordered_map<std::string, MLValue>& feeds,
                                       std::vector<std::string>& output_names) {
  (input_defs);
  for (auto& elem : output_defs) {
    output_names.push_back(elem->Name());
  }

  for (auto& input : input_data_) {
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

void OpTester::Run(ExpectResult expect_result, const std::string& expected_failure_string) {
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

    node.SetExecutionProvider(provider_name_);
    Status status = graph->Resolve();
    LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());

    // Hookup the inputs and outputs
    std::unordered_map<std::string, MLValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

    // Run the model
    SessionOptions so;
    so.session_logid = op_;
    so.session_log_verbosity_level = 1;

    IExecutionProvider* p_provider = provider_.release();  // the ownership is taken by session_object, but we kept this for tensor copy during session
    InferenceSession session_object{so};
    if (p_provider)
      EXPECT_TRUE(session_object.RegisterExecutionProvider(std::unique_ptr<IExecutionProvider>(p_provider)).IsOK());

    status = session_object.Load(std::move(p_model));
    EXPECT_TRUE(status.IsOK());
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Load failed with error: " << status.ErrorMessage();
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
    if (!status.IsOK()) {
      if (expect_result == ExpectResult::kExpectFailure) {
        EXPECT_TRUE(!status.IsOK());
        EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
      } else {
        LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
        EXPECT_TRUE(status.IsOK());
      }
    }
    if (!status.IsOK()) {
      return;
    }

    // Verify the outputs
    // Todo: support check output with map/sequence/....
    int idx = 0;
    for (auto& expected_data : output_data_) {
      MLValue& mlvalue = fetches[idx];
      if (expected_data.data_.IsTensor()) {
        if (provider_name_ == LotusIR::kCudaExecutionProvider) {
          //TODO: remove this workaround once CUDATransform adds copy from GPU, assuming float tensor
          auto& gpu_tensor = mlvalue.Get<Tensor>();
          auto& cpu_arena = AllocatorManager::Instance().GetArena(CPU);
          auto bytes = gpu_tensor.Shape().Size() * gpu_tensor.DataType()->Size();
          void* p = cpu_arena.Alloc(bytes);
          Tensor cpu_tensor(gpu_tensor.DataType(), gpu_tensor.Shape(), p, cpu_arena.Info());
          p_provider->CopyTensor(gpu_tensor, cpu_tensor);
          Check(expected_data, cpu_tensor);
          cpu_arena.Free(p);
        } else
          Check(expected_data, mlvalue.Get<Tensor>());
      } else {
        Check(expected_data, mlvalue);
      }
      ++idx;
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    // rethrow as some tests for error handling expect this
    throw;
  }
}

template <typename T>
void OpTester::AddData(std::vector<Data>& data, const char* name,
                       const std::vector<int64_t>& dims, const T* values,
                       int64_t valuesCount, bool on_cpu) {
  LOTUS_ENFORCE(TensorShape(dims).Size() == valuesCount, "Number of input values doesn't match tensor size");

  //TODO: temporary workaround before CUDA graph transform adds copy node
  on_cpu = on_cpu || (provider_name_ == LotusIR::kCpuExecutionProvider);
  auto& allocator = AllocatorManager::Instance().GetAllocator(on_cpu ? CPU : CUDA, 0, on_cpu); // use device allocator for CUDA inputs
  auto size_in_bytes = valuesCount * sizeof(T);
  void* buffer = allocator.Alloc(size_in_bytes);
  auto p_tensor = make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                      TensorShape(dims),
                                      buffer,
                                      allocator.Info(),
                                      &allocator);
  auto* data_ptr = p_tensor->template MutableData<T>();

  if (on_cpu) {
    for (int64_t i = 0; i < valuesCount; i++) {
      data_ptr[i] = values[i];
    }
  } else {
    Tensor cpu_tensor(DataTypeImpl::GetType<T>(),
                      TensorShape(dims),
                      const_cast<T*>(values),
                      AllocatorInfo("CPU_inplace", kDeviceAllocator));
    provider_->CopyTensor(cpu_tensor, *p_tensor);
  }

  MLValue value;
  value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  data.push_back({{name, &s_type_proto<T>}, value});
}  // namespace Test

#define OPTEST_ADDDATA_TEMPLATE(T)                                                      \
  template void OpTester::AddData<T>(std::vector<Data> & data, const char* name,        \
                                     const std::vector<int64_t>& dims, const T* values, \
                                     int64_t valuesCount, bool on_cpu)

OPTEST_ADDDATA_TEMPLATE(float);
OPTEST_ADDDATA_TEMPLATE(double);
OPTEST_ADDDATA_TEMPLATE(int8_t);
OPTEST_ADDDATA_TEMPLATE(uint8_t);
OPTEST_ADDDATA_TEMPLATE(int16_t);
OPTEST_ADDDATA_TEMPLATE(uint16_t);
OPTEST_ADDDATA_TEMPLATE(int32_t);
OPTEST_ADDDATA_TEMPLATE(uint32_t);
OPTEST_ADDDATA_TEMPLATE(int64_t);
OPTEST_ADDDATA_TEMPLATE(uint64_t);
OPTEST_ADDDATA_TEMPLATE(bool);
OPTEST_ADDDATA_TEMPLATE(std::string);

}  // namespace Test
}  // namespace Lotus
