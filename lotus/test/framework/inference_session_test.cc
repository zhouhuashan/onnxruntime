#include "core/framework/inference_session.h"

#include <functional>
#include <thread>
#include "core/common/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "gtest/gtest.h"

using namespace std;

namespace Lotus {
namespace Test {

static const std::string MODEL_URI = "testdata/mul_1.pb";

// TODO consider moving this function to some utils
template <typename T>
void CreateMLValue(IAllocator* alloc,
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

  Tensor* tensor = new Tensor(
      element_type,
      shape,
      std::move(BufferUniquePtr(buffer, BufferDeleter(alloc))),
      location);
  p_mlvalue->Init(tensor,
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

void RunModel(InferenceSession& session_object, const RunOptions& run_options) {
  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value;
  CreateMLValue<float>(&AllocatorManager::Instance()->GetArena(CPU), dims_mul_x, values_mul_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Now run
  Common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);

  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims_mul_y);
  EXPECT_EQ(expected_shape, rtensor.shape());
  const std::vector<float> found(rtensor.data<float>(), rtensor.data<float>() + expected_shape.Size());
  ASSERT_EQ(expected_values_mul_y, found);
}

TEST(InferenceSessionTestNoTimeout, RunTest) {
  ExecutionProviderInfo epi;
  ProviderOption po{"CPUExecutionProvider", epi};
  SessionOptions so(vector<ProviderOption>{po}, true);

  InferenceSession session_object{so};
  EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  RunModel(session_object, run_options);
}

TEST(MultipleInferenceSessionTestNoTimeout, RunTest) {
  ExecutionProviderInfo epi;
  ProviderOption po{"CPUExecutionProvider", epi};
  SessionOptions session_options(vector<ProviderOption>{po}, true);
  session_options.ep_options.push_back(po);
  InferenceSession session_object{session_options};
  EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  std::thread thread1{[&session_object]() {
    RunOptions run_options;
    run_options.run_tag = "one session/thread 1";
    RunModel(session_object, run_options);
  }};

  std::thread thread2{[&session_object]() {
    RunOptions run_options;
    run_options.run_tag = "one session/thread 2";
    RunModel(session_object, run_options);
  }};

  thread1.join();
  thread2.join();
}

// TODO write test with timeout

}  // namespace Test
}  // namespace Lotus
