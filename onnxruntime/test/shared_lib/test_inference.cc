#include "core/session/inference_session.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/environment.h"
#include "core/framework/ml_value.h"

using namespace std;
using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::Logging;

static const std::string MODEL_URI = "testdata/mul_1.pb";
static const std::string CUSTOM_OP_MODEL_URI = "testdata/foo_1.pb";

template <typename T>
void CreateMLValue(AllocatorPtr alloc,
                   const std::vector<int64_t>& dims,
                   const std::vector<T>& value,
                   MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto location = alloc->Info();
  auto element_type = DataTypeImpl::GetType<T>();
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  if (value.size() > 0) {
    memcpy(buffer, &value[0], element_type->Size() * shape.Size());
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location,
                                                              alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

void RunSession(InferenceSession& session_object,
                RunOptions& run_options,
                AllocatorPtr alloc,
                const std::vector<int64_t>& dims_x,
                const std::vector<float>& values_x,
                const std::vector<int64_t>& dims_y,
                const std::vector<float>& values_y) {
  // prepare inputs
  MLValue ml_value;
  CreateMLValue<float>(alloc, dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "error in Run() " << st.ErrorMessage() << std::endl;
    exit(1);
  }
  if (1 != fetches.size()) {
    std::cout << "output sizes don't match: 1 != " << fetches.size() << std::endl;
    exit(1);
  }
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  if (expected_shape != rtensor.Shape()) {
    std::cout << "shapes don't match: expected_shape " << expected_shape << " != rtensor.Shape() " << rtensor.Shape() << std::endl;
    exit(1);
  }
  const std::vector<float> found(rtensor.Data<float>(), rtensor.Data<float>() + expected_shape.Size());
  if (values_y != found) {
    std::cout << "outputs don't match" << std::endl;
    exit(1);
  }
  cout << "Run() succeeded\n";
}

class CapturingSink : public Logging::ISink {
 public:
  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override {
    // operator for formatting of timestamp in ISO8601 format including microseconds
    //using date::operator<<;
    UNUSED_PARAMETER(timestamp);
    std::cout << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
              << message.Location().ToString() << "] " << message.Message() << std::endl;
  }
};

void TestInference(const std::string& model_uri,
                   const std::vector<int64_t>& dims_x,
                   const std::vector<float>& values_x,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   bool custom_op) {
  static std::string default_logger_id{"TestSharedLib"};
  auto logging_manager = std::make_unique<Logging::LoggingManager>(
      std::unique_ptr<ISink>(new CapturingSink()), Logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Default, &default_logger_id);

  SessionOptions so;
  so.session_logid = default_logger_id;
  InferenceSession session_object{so, logging_manager.get()};

  if (custom_op) {
    auto st = session_object.LoadCustomOps({"liblotus_custom_op_shared_lib_test.so"});
    if (!st.IsOK()) {
      std::cout << "error loading custom ops library " << st.ErrorMessage() << std::endl;
      exit(1);
    }
  }

  Status st = session_object.Load(model_uri);
  if (!st.IsOK()) {
    std::cout << "error loading model " << st.ErrorMessage() << std::endl;
    exit(1);
  }
  st = session_object.Initialize();
  if (!st.IsOK()) {
    std::cout << "error initializing " << st.ErrorMessage() << std::endl;
    exit(1);
  }

  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  AllocatorPtr cpu_allocator = std::make_shared<::onnxruntime::CPUAllocator>();
  // Now run
  RunSession(session_object, run_options, cpu_allocator, dims_x, values_x, expected_dims_y, expected_values_y);
}

int main() {
  std::unique_ptr<Environment> test_environment;
  Status status = Environment::Create(test_environment);
  if (!status.IsOK()) {
    cout << "error creating environment with error " << status.ErrorMessage() << endl;
    exit(1);
  }

  {
    // simple inference test
    // prepare inputs
    std::cout << "Running simple inference" << std::endl;
    std::vector<int64_t> dims_x = {3, 2};
    std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // prepare expected inputs and outputs
    std::vector<int64_t> expected_dims_y = {3, 2};
    std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

    TestInference(MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, false);
  }

  // custom op test
  // prepare inputs
  {
    std::cout << "Running custom op inference" << std::endl;
    std::vector<int64_t> dims_x = {3, 2};
    std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // prepare expected inputs and outputs
    std::vector<int64_t> expected_dims_y = {3, 2};
    std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

    TestInference(CUSTOM_OP_MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, true);
  }

  return 0;
}
