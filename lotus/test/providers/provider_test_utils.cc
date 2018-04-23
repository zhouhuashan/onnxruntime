#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"

#include <exception>
#include <memory>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

using namespace Lotus::Logging;

namespace Lotus {
namespace Test {

// These have to be defined before Run() since Run() tries to instantiate them
// The Check templates have to be defined before Run() since Run() instantiates them

// The default implementation compares for equality, specialized versions for other types are below
template <typename T>
void OpTester::Check(const Data& output_data, const Tensor& output_tensor, size_t size) {
  auto& expected_tensor = output_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.Data<T>();
  auto* output = output_tensor.Data<T>();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

template <>
void OpTester::Check<float>(const Data& output_data, const Tensor& output_tensor, size_t size) {
  auto& expected_tensor = output_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.Data<float>();
  auto* output = output_tensor.Data<float>();
  bool has_abs_err = output_data.absolute_error_.has_value();
  bool has_rel_err = output_data.relative_error_.has_value();
  for (int i = 0; i < size; ++i) {
    if (std::isinf(expected[i]))  // Test infinity for equality
      EXPECT_EQ(expected[i], output[i]);
    else {
      if (!has_abs_err && !has_rel_err) {
        // the default for existing tests
        EXPECT_NEAR(expected[i], output[i], 0.001f);
      } else {
        if (has_abs_err) {
          EXPECT_NEAR(expected[i], output[i], output_data.absolute_error_.value());
        }
        if (has_rel_err) {
          EXPECT_NEAR(expected[i], output[i], output_data.relative_error_.value() * std::abs(expected[i]));
        }
      }
    }
  }
}

OpTester::~OpTester() {
#if _DEBUG
  if (!run_called_) {
    std::cerr << "Someone forgot to call OpTester::Run()" << std::endl;
    __debugbreak();
  }
#endif
}

void OpTester::CreateMLValue(IAllocator* alloc,
                             const std::vector<int64_t>& dims,
                             MLDataType element_type,
                             const gsl::byte* p_value,
                             size_t input_size_bytes,
                             MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto location = alloc->Info();
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  LOTUS_ENFORCE(p_value);
  memcpy(buffer, p_value, input_size_bytes);

  auto p_tensor = std::make_unique<Tensor>(element_type, shape, buffer, location, alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
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

void OpTester::Run(bool expect_failure, const std::string& expected_failure_string) {
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

    node.SetExecutionProvider(LotusIR::kCpuExecutionProvider);
    Status status = graph->Resolve();
    LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());

    // Hookup the inputs and outputs
    std::unordered_map<std::string, MLValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

    // Run the model
    ExecutionProviderInfo epi;
    ProviderOption po{LotusIR::kCpuExecutionProvider, epi};
    SessionOptions so(vector<ProviderOption>{po});
    so.session_logid = op_;
    so.session_log_verbosity_level = 1;

    InferenceSession session_object{so};
    status = session_object.Load(std::move(p_model));
    EXPECT_TRUE(status.IsOK());
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Load failed with error: " << status.ErrorMessage();
      return;
    }

    status = session_object.Initialize();
    EXPECT_TRUE(status.IsOK());
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Initialize failed with error: " << status.ErrorMessage();
      return;
    }

    RunOptions run_options;
    run_options.run_tag = op_;
    run_options.run_log_verbosity_level = 1;
    std::vector<MLValue> fetches;
    status = session_object.Run(run_options, feeds, output_names, &fetches);
    if (expect_failure) {
      EXPECT_TRUE(!status.IsOK());
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
    } else {
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
      }
      EXPECT_TRUE(status.IsOK());
    }

    if (!status.IsOK()) {
      return;
    }

    // Verify the outputs
    // Todo: support check output with map/sequence/....
    int idx = 0;
    for (auto& output : output_data_) {
      LOTUS_ENFORCE(output.data_.IsTensor());
      auto& output_tensor = output.data_.Get<Tensor>();
      MLValue& mlvalue = fetches[idx];
      auto& result_tensor = mlvalue.Get<Tensor>();
      LOTUS_ENFORCE(output_tensor.Shape() == result_tensor.Shape(), "Output shape did not match expected output shape");
      auto size = output_tensor.Shape().Size();

      // Dispatch on the type
      auto type = output_tensor.DataType();

      if (type == DataTypeImpl::GetType<float>()) {
        Check<float>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<bool>()) {
        Check<bool>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<int64_t>()) {
        Check<int64_t>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<double>()) {
        Check<double>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<uint8_t>()) {
        Check<uint8_t>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<uint16_t>()) {
        Check<uint16_t>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<uint32_t>()) {
        Check<uint32_t>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<uint64_t>()) {
        Check<uint64_t>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<int16_t>()) {
        Check<int16_t>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<int32_t>()) {
        Check<int32_t>(output, result_tensor, size);
      } else if (type == DataTypeImpl::GetType<std::string>()) {
        Check<std::string>(output, result_tensor, size);
      } else {
        LOTUS_THROW("OpTester:Check() not implemented for output tensor type of ", type);
      }
      ++idx;
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    // rethrow as some tests for error handling expect this
    throw;
  }
}  // namespace Test

}  // namespace Test
}  // namespace Lotus
