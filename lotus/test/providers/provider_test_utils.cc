#include "test/providers/provider_test_utils.h"

#include <memory>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "test/test_utils.h"

using namespace Lotus::Logging;

namespace Lotus {
namespace Test {

// These have to be defined before Run() since Run() tries to instantiate them
template <>
void OpTester::Check<float>(const Data& output_data, const Tensor& output_tensor, size_t size) {
  auto* expected = reinterpret_cast<const float*>(output_data.data_.get());
  auto* output = output_tensor.Data<float>();
  for (int i = 0; i < size; ++i) {
    if (std::isinf(expected[i]))  // Test infinity for equality
      EXPECT_EQ(expected[i], output[i]);
    else
      EXPECT_NEAR(expected[i], output[i], 0.001f);
  }
}

template <>
void OpTester::Check<bool>(const Data& output_data, const Tensor& output_tensor, size_t size) {
  auto* expected = reinterpret_cast<const bool*>(output_data.data_.get());
  auto* output = output_tensor.Data<bool>();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(expected[i], output[i]);
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

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type, shape, buffer, location, alloc);
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
    MLValue mlvalue;
    CreateMLValue(&AllocatorManager::Instance().GetArena(CPU),
                  input.shape_.GetDims(),
                  input.data_type_,
                  input.data_.get(),
                  input.data_size_in_bytes_,
                  &mlvalue);
    feeds[input.def_.Name()] = mlvalue;
  }
}

void OpTester::Run(bool expect_failure) {
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
  std::unique_ptr<LotusIR::Model> p_model = std::make_unique<LotusIR::Model>("test");
  LotusIR::Graph* graph = p_model->MainGraph();
  graph->AddNode("node1", op_, op_, input_defs, output_defs);
  graph->Resolve();

  auto& node = *graph->GetNode(graph->NumberOfNodes() - 1);

  // Add the attributes if any
  for (auto& add_attribute_fn : add_attribute_funcs_)
    add_attribute_fn(node);

  node.SetExecutionProvider(LotusIR::kCpuExecutionProvider);

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
  Common::Status status = session_object.Load(std::move(p_model));
  EXPECT_TRUE(status.IsOK());
  if (!status.IsOK()) {
    LOGS_DEFAULT(INFO) << "Load failed with error: " << status.ErrorMessage();
    return;
  }

  status = session_object.Initialize();
  EXPECT_TRUE(status.IsOK());
  if (!status.IsOK()) {
    LOGS_DEFAULT(INFO) << "Initialize failed with error: " << status.ErrorMessage();
    return;
  }

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;
  std::vector<MLValue> fetches;
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  if (expect_failure) {
    EXPECT_TRUE(!status.IsOK());
  } else {
    EXPECT_TRUE(status.IsOK());
  }
  if (!status.IsOK()) {
    LOGS_DEFAULT(INFO) << "Run failed with error: " << status.ErrorMessage();
    return;
  }

  // Verify the outputs
  int idx = 0;
  for (auto& output : output_data_) {
    MLValue& mlvalue = fetches[idx];
    auto& result_tensor = mlvalue.Get<Tensor>();
    LOTUS_ENFORCE(output.shape_ == result_tensor.Shape(), "Output shape did not match expected output shape");
    auto size = output.shape_.Size();

    // Dispatch on the type
    if (output.data_type_ == DataTypeImpl::GetType<float>()) {
      Check<float>(output, result_tensor, size);
    } else if (output.data_type_ == DataTypeImpl::GetType<bool>()) {
      Check<bool>(output, result_tensor, size);
    }
    ++idx;
  }
}

}  // namespace Test
}  // namespace Lotus
