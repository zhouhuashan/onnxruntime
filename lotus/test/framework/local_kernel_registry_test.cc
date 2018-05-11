#include "core/framework/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "core/inc/op_kernel_author_helper.h"

#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
// Foo kernel which is doing Add
template <typename T>
class FooKernel {
 public:
  FooKernel(const MLOpKernelInfo& /*info*/) {}

  MLStatus Compute(const MLOpKernelInfo& /*info*/, const MLOpKernelContext& context) const {
    const auto X = context.GetInputTensor(0);
    const auto W = context.GetInputTensor(1);

    auto X_Data = X.GetData<T>();
    auto W_Data = W.GetData<T>();

    auto& shape = X.GetDimensions();
    auto Y = context.GetOutputTensor(0, shape);
    auto Y_Data = Y.GetData<T>();

    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      Y_Data[i] = X_Data[i] + W_Data[i];
    }

    return MLStatus::OK;
  }
};

//For test purpose, we register this Foo kernel to Mul op.
//Once the custom schema is ready, should update this.
KernelDefBuilder FooKernelDef() {
  KernelDefBuilder def("Mul");
  def.Domain(LotusIR::kOnnxDomain)
      .SinceVersion(1)
      .Provider(LotusIR::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

MLStatus CreateFooKernel(const IMLOpKernelInfo& kernelInfo, IMLOpKernel** opKernel) {
  return MLOpKernel<FooKernel<float> >::CreateInstance(kernelInfo, opKernel);
}

static const std::string MODEL_URI = "testdata/mul_1.pb";

TEST(CustomKernelTests, CustomKernel) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  auto def = FooKernelDef();
  //Register a foo kernel which is doing Add, but bind to Mul.
  EXPECT_TRUE(session_object.RegisterCustomKernel(def, CreateFooKernel).IsOK());
  EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(), dims_mul_x, values_mul_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_mul_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  Common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims_mul_y);
  EXPECT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.Data<float>(), rtensor.Data<float>() + expected_shape.Size());
  ASSERT_EQ(expected_values_mul_y, found);
}
}  // namespace Test
}  // namespace Lotus
