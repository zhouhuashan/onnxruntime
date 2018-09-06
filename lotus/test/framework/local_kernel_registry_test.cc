#include "core/session/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_abi_wrapper.h"
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
#include "core/graph/schema_registry.h"
#include "core/framework/customregistry.h"
using namespace onnx;
namespace Lotus {
namespace Test {

// Checks test attributes set on ABI kernels can be queried with correct values
void VerifyTestAttributes(const MLOperatorAttributes& attrs) {
  std::string str_attr = attrs.GetAttribute("DefaultedNonRequiredString");
  ASSERT_EQ(str_attr, "1");

  std::vector<std::string> str_array_attr = attrs.GetAttributeVector("DefaultedNonRequiredStringArray");
  std::vector<std::string> expected = std::vector<std::string>({"1", "2"});
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(str_array_attr[i], expected[i]);
  }

  ASSERT_EQ(1, attrs.GetAttribute<int64_t>("DefaultedNonRequiredInt"));
  ASSERT_EQ(1.0f, attrs.GetAttribute<float>("DefaultedNonRequiredFloat"));

  ASSERT_EQ(std::vector<int64_t>({1, 2}), attrs.GetAttributeVector<int64_t>("DefaultedNonRequiredIntArray"));
  ASSERT_EQ(std::vector<float>({1.0f, 2.0f}), attrs.GetAttributeVector<float>("DefaultedNonRequiredFloatArray"));
}

// Foo kernel which is doing Add
template <typename T, bool VerifyAttributes = false, bool Truncate = false>
class FooKernel {
 public:
  FooKernel(const MLOpKernelInfo& info) {
    if (VerifyAttributes) {
      VerifyTestAttributes(info);
    }

    VerifyShapeInfo(info);
  }

  void VerifyShapeInfo(const MLOpKernelInfo& info) {
    if (!Truncate) {
      const IMLOpKernelTensorShapeInfo* shape_info;
      ASSERT_EQ(info.GetInterface()->HasTensorShapeInfo(), false);
      ASSERT_EQ(info.GetInterface()->GetTensorShapeInfo(&shape_info), MLStatus::REQUIREMENT_NOT_REGISTERED);
    } else {
      const IMLOpKernelTensorShapeInfo* shape_info;
      ASSERT_EQ(info.GetInterface()->HasTensorShapeInfo(), true);
      ASSERT_EQ(info.GetInterface()->GetTensorShapeInfo(&shape_info), MLStatus::OK);
    }
  }

  void Compute(const MLOpKernelContext& context) const {
    const auto X = context.GetInputTensor(0);
    const auto W = context.GetInputTensor(1);

    auto X_Data = X.GetData<T>();
    auto W_Data = W.GetData<T>();

    auto shape = X.GetDimensions();

    // This is used to test shape inference
    if (Truncate) {
      shape[0] -= 1;
    }
    if (!Truncate) {
      IMLOpTensor* tensor;
      ASSERT_EQ(context.GetInterface()->GetOutputTensor(0, &tensor), MLStatus::SHAPE_INFERENCE_NOT_REGISTERED);
    } else {
      IMLOpTensor* tensor;
      ASSERT_EQ(context.GetInterface()->GetOutputTensor(0, &tensor), MLStatus::OK);
    }
    auto Y = context.GetOutputTensor(0, shape);
    auto Y_Data = Y.GetData<T>();

    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      Y_Data[i] = X_Data[i] + W_Data[i];
    }
  }
};

template <typename T>
class OptionalOpKernel {
 public:
  OptionalOpKernel(const MLOpKernelInfo& /*info*/) {}

  void Compute(const MLOpKernelContext& context) const {
    const auto X = context.GetInputTensor(0);
    const auto W = context.GetInputTensor(1);

    auto X_Data = X.GetData<T>();
    auto& shape = X.GetDimensions();
    auto Y = context.GetOutputTensor(0, shape);
    auto Y_Data = Y.GetData<T>();
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      Y_Data[i] = X_Data[i];
    }

    auto Y2 = context.GetOutputTensor(1, shape);
    // Y2 is used or not
    if (!Y2.IsUnused()) {
      auto Y2_Data = Y2.GetData<T>();
      for (size_t i = 0; i < size; i++) {
        Y2_Data[i] = X_Data[i];
      }
    }

    //W is used or not
    if (!W.IsUnused()) {
      auto W_Data = W.GetData<T>();
      for (size_t i = 0; i < size; i++) {
        Y_Data[i] += W_Data[i];
      }
      if (!Y2.IsUnused()) {
        auto Y2_Data = Y2.GetData<T>();
        for (size_t i = 0; i < size; i++) {
          Y2_Data[i] += W_Data[i];
        }
      }
    }
  }
};

onnx::OpSchema GetFooSchema() {
  onnx::OpSchema schema("Foo", "unknown", 0);
  schema.Input(0,
               "A",
               "First operand, should share the type with the second operand.",
               "T");
  schema.Input(
      1,
      "B",
      "Second operand. With broadcasting can be of smaller size than A. "
      "If broadcasting is disabled it should be of the same size.",
      "T");
  schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
  schema.TypeConstraint(
      "T",
      OpSchema::numeric_types_for_math_reduction(),
      "Constrain input and output types to high-precision numeric tensors.");
  schema.SinceVersion(7);
  return schema;
}

onnx::OpSchema GetOptionalOpSchema() {
  onnx::OpSchema schema("OptionalOp", "unknown", 0);
  schema.Input(0,
               "X",
               "First operand, should share the type with the second operand.",
               "T");
  schema.Input(
      1,
      "W",
      "Second operand. If provided, add it to the output",
      "T",
      OpSchema::Optional);
  schema.Output(0, "Y", "Result, has same dimensions and type as A", "T");
  schema.Output(1, "Y2", "Result, has same dimensions and type as A", "T", OpSchema::Optional);
  schema.TypeConstraint(
      "T",
      OpSchema::numeric_types_for_math_reduction(),
      "Constrain input and output types to high-precision numeric tensors.");
  schema.SinceVersion(6);
  return schema;
}

//For test purpose, we register this Foo kernel to Mul op.
//Once the custom schema is ready, should update this.
KernelDefBuilder FooKernelDef(const char* schema_name) {
  KernelDefBuilder def;
  def.SetName(schema_name)
      .SetDomain(LotusIR::kOnnxDomain)
      .SinceVersion(7)
      .Provider(LotusIR::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

// Creates a Foo kernel implementing the ABI
template <bool VerifyTestAttributes = false>
MLStatus CreateABIFooKernel(const IMLOpKernelInfo& kernel_info, IMLOpKernel** op_kernel) {
  return MLOpKernel<FooKernel<float, VerifyTestAttributes>>::CreateInstance(kernel_info, op_kernel);
}

template <bool VerifyTestAttributes = false>
MLStatus CreateABIOptionalKernel(const IMLOpKernelInfo& kernel_info, IMLOpKernel** op_kernel) {
  return MLOpKernel<OptionalOpKernel<float>>::CreateInstance(kernel_info, op_kernel);
}

KernelDefBuilder OptionalKernelDef() {
  KernelDefBuilder def;
  def.SetName("OptionalOp")
      .SetDomain(LotusIR::kOnnxDomain)
      .SinceVersion(6)
      .Provider(LotusIR::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

// Creates a Foo kernel implementing the ABI
MLStatus CreateTruncatedABIFooKernel(const IMLOpKernelInfo& kernel_info, IMLOpKernel** op_kernel) {
  return MLOpKernel<FooKernel<float, true, true>>::CreateInstance(kernel_info, op_kernel);
}

// Creates a Foo kernel implementing the built-in OpKernel type.  This wraps
// the ABI kernel as an implementation detail.
OpKernel* CreateFooKernel(const OpKernelInfo& kernel_info) {
  return new ::Lotus::AbiOpKernel(CreateABIFooKernel, kernel_info, false, false, nullptr, nullptr);
}

OpKernel* CreateOptionalOpKernel(const OpKernelInfo& kernel_info) {
  return new ::Lotus::AbiOpKernel(CreateABIOptionalKernel, kernel_info, false, false, nullptr, nullptr);
}
static const std::string MUL_MODEL_URI = "testdata/mul_1.pb";
static const std::string FOO_MODEL_URI = "testdata/foo_1.pb";
static const std::string FOO_TRUNCATE_MODEL_URI = "testdata/foo_2.pb";

static const std::string OPTIONAL_MODEL1_URI = "testdata/optional_1.pb";

void RunSession(InferenceSession& session_object,
                RunOptions& run_options,
                std::vector<int64_t>& dims_x,
                std::vector<float>& values_x,
                std::vector<int64_t>& dims_y,
                std::vector<float>& values_y) {
  // prepare inputs
  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(kMemTypeDefault), dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // Now run
  Common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  EXPECT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.Data<float>(), rtensor.Data<float>() + expected_shape.Size());
  ASSERT_EQ(values_y, found);
}

TEST(CustomKernelTests, CustomKernelWithBuildInSchema) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  // Register a foo kernel which is doing Add, but bind to Mul.
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
  auto def = FooKernelDef("Mul");

  EXPECT_TRUE(registry->RegisterCustomKernel(def, CreateFooKernel).IsOK());

  EXPECT_TRUE(session_object.Load(MUL_MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}

TEST(CustomKernelTests, CustomABIKernelWithBuildInSchema) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  AbiCustomRegistry abi_registry(registry);

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

  //Register a foo kernel which is doing Add, but bind to Mul.
  MLEdgeType floatTensorType = {
      MLEdgeClass::kTensor,
      MLTensorDataType::kFloat};

  MLTypeConstraint constraint = {"T", &floatTensorType, 1};

  MLOpKernelDefinition def = {
      LotusIR::kOnnxDomain,
      "Mul",
      7,
      LotusIR::kCpuExecutionProvider,
      &constraint,
      1,
      nullptr,
      0,
      nullptr,
      nullptr};

  EXPECT_EQ(MLStatus::OK, abi_registry.RegisterOpKernel(&def, MLOpKernelOptions::kAllowDynamicInputShapes, CreateABIFooKernel));

  EXPECT_TRUE(session_object.Load(MUL_MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}

TEST(CustomKernelTests, CustomKernelWithCustomSchema) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

  //register foo schema
  auto foo_schema = GetFooSchema();
  std::vector<OpSchema> schemas = {foo_schema};
  EXPECT_TRUE(registry->RegisterOpSet(schemas, LotusIR::kOnnxDomain, 5, 7).IsOK());
  auto def = FooKernelDef("Foo");
  //Register a foo kernel which is doing Add, but bind to Mul.
  EXPECT_TRUE(registry->RegisterCustomKernel(def, CreateFooKernel).IsOK());

  EXPECT_TRUE(session_object.Load(FOO_MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}

TEST(CustomKernelTests, CustomABIKernelWithCustomABISchema) {
  // Test cases
  struct {
    bool type_label;
    bool type_inf;
    bool shape_inf;
  } test_cases[] = {
      {true, false, false},
      {false, false, false},
      {true, true, false},
      {false, false, true},
      {true, true, true},
  };

  for (int case_index = 0; case_index < sizeof(test_cases) / sizeof(test_cases[0]); ++case_index) {
    SessionOptions so;

    so.session_logid = "InferenceSessionTests.NoTimeout";

    std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
    AbiCustomRegistry abi_registry(registry);

    InferenceSession session_object{so, &DefaultLoggingManager()};
    EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

    // Input and output parameters
    MLFormalParameter input_param = {MLFormalParameterOptions::kSingle, MLFormalParameterTypeFormat::kLabel, "T"};
    if (!test_cases[case_index].type_label) {
      assert(!test_cases[case_index].type_inf);
      input_param.type_format = MLFormalParameterTypeFormat::kEdgeType;
      input_param.edge_type.edge_class = MLEdgeClass::kTensor;
      input_param.edge_type.tensor_data_type = MLTensorDataType::kFloat;
    } else {
      input_param.type_label = "T1";
    }

    MLFormalParameter output_param = input_param;

    // Type inference should set this to tensor(float) even though T2 is not matched
    // on an input label
    if (test_cases[case_index].type_inf) {
      output_param.type_label = "T2";
      output_param.edge_type.tensor_data_type = MLTensorDataType::kInt32;
    }

    MLFormalParameter inputs[] = {input_param, input_param};
    MLEdgeType edgeTypes[] = {{MLEdgeClass::kTensor, MLTensorDataType::kUInt32},
                              {MLEdgeClass::kTensor, MLTensorDataType::kUInt64},
                              {MLEdgeClass::kTensor, MLTensorDataType::kInt32},
                              {MLEdgeClass::kTensor, MLTensorDataType::kInt64},
                              {MLEdgeClass::kTensor, MLTensorDataType::kFloat},
                              {MLEdgeClass::kTensor, MLTensorDataType::kDouble}};

    MLTypeConstraint constraints[] = {
        {"T1", edgeTypes, sizeof(edgeTypes) / sizeof(edgeTypes[0])},
        {"T2", edgeTypes, sizeof(edgeTypes) / sizeof(edgeTypes[0])}};

    // Test attributes
    MLAttribute attributes[] = {
        {"DefaultedNonRequiredInt", MLAttributeType::kInt, false},
        {"DefaultedNonRequiredFloat", MLAttributeType::kFloat, false},
        {"DefaultedNonRequiredString", MLAttributeType::kString, false},
        {"DefaultedNonRequiredIntArray", MLAttributeType::kIntArray, false},
        {"DefaultedNonRequiredFloatArray", MLAttributeType::kFloatArray, false},
        {"DefaultedNonRequiredStringArray", MLAttributeType::kStringArray, false},

        {"NonDefaultedNonRequiredStringArray", MLAttributeType::kStringArray, false},
    };

    // Defaults.  These are queried back during kernel creation, type and shape inference
    // and tested against the same values
    MLAttributeNameValue default_attributes[] = {
        {"DefaultedNonRequiredInt", MLAttributeType::kInt, 1, {}},
        {"DefaultedNonRequiredFloat", MLAttributeType::kFloat, 1, {}},
        {"DefaultedNonRequiredString", MLAttributeType::kString, 1, {}},
        {"DefaultedNonRequiredIntArray", MLAttributeType::kIntArray, 2, {}},
        {"DefaultedNonRequiredFloatArray", MLAttributeType::kFloatArray, 2, {}},
        {"DefaultedNonRequiredStringArray", MLAttributeType::kStringArray, 2, {}},
    };

    int64_t default_ints[] = {1, 2};
    float default_floats[] = {1.0f, 2.0f};
    const char* default_strings[] = {"1", "2"};
    default_attributes[0].ints = default_ints;
    default_attributes[1].floats = default_floats;
    default_attributes[2].strings = default_strings;
    default_attributes[3].ints = default_ints;
    default_attributes[4].floats = default_floats;
    default_attributes[5].strings = default_strings;

    // Schema definition
    MLSchemaDefinition def = {};
    def.name = "Foo";
    def.operator_set_since_version = 7;
    def.inputs = inputs;
    def.input_count = 2;
    def.outputs = &output_param;
    def.output_count = 1;
    def.type_constraints = constraints;
    def.type_constraint_count = test_cases[case_index].type_label ? 2 : 0;
    def.attributes = attributes;
    def.attribute_count = sizeof(attributes) / sizeof(attributes[0]);
    def.default_attributes = default_attributes;
    def.default_attribute_count = sizeof(default_attributes) / sizeof(default_attributes[0]);

    // Type inference function
    if (test_cases[case_index].type_inf) {
      def.type_inference_function_context = (void*)123;
      def.type_inference_function = [](void* reg_ctx, IMLTypeInferenceContext* ctx) -> MLStatus {
        EXPECT_EQ(reg_ctx, (void*)123);
        VerifyTestAttributes(MLTypeInferenceContext(ctx));
        MLEdgeType output_type = {MLEdgeClass::kTensor, MLTensorDataType::kFloat};
        MLTypeInferenceContext(ctx).SetOutputEdgeType(0, &output_type);
        return MLStatus::OK;
      };
    }

    // Shape inference is tested by truncating the output size
    bool truncate_output = test_cases[case_index].shape_inf;
    if (truncate_output) {
      def.shape_inference_function_context = (void*)456;
      def.shape_inference_function = [](void* reg_ctx, IMLShapeInferenceContext* ctx) -> MLStatus {
        EXPECT_EQ(reg_ctx, (void*)456);
        VerifyTestAttributes(MLShapeInferenceContext(ctx));
        MLShapeInferenceContext(ctx).SetOutputTensorShape(0, {2, 2});
        return MLStatus::OK;
      };
    }

    // Register the schema
    MLOperatorSetId id = {"", 7};
    MLSchemaDefinition* def_list = &def;
    EXPECT_EQ(MLStatus::OK, abi_registry.RegisterOpSetFromSchema(&id, 1, &def_list, 1));

    // Register a foo kernel which is doing Add, but bind to Mul.
    MLEdgeType floatTensorType = {
        MLEdgeClass::kTensor,
        MLTensorDataType::kFloat};

    MLTypeConstraint kernel_constraint = {"T", &floatTensorType, 1};

    MLOpKernelDefinition kernel_def = {
        LotusIR::kOnnxDomain,
        "Foo",
        7,
        LotusIR::kCpuExecutionProvider,
        &kernel_constraint,
        1,
        nullptr,
        0,
        def.shape_inference_function,
        def.shape_inference_function_context};

    if (!truncate_output) {
      MLOpKernelOptions options = MLOpKernelOptions::kAllowDynamicInputShapes;
      EXPECT_EQ(MLStatus::OK, abi_registry.RegisterOpKernel(&kernel_def, options, CreateABIFooKernel<true>));
    } else {
      MLOpKernelOptions options = MLOpKernelOptions::kNone;
      EXPECT_EQ(MLStatus::OK, abi_registry.RegisterOpKernel(&kernel_def, options, CreateTruncatedABIFooKernel));
    }

    EXPECT_TRUE(session_object.Load(truncate_output ? FOO_TRUNCATE_MODEL_URI : FOO_MODEL_URI).IsOK());
    EXPECT_TRUE(session_object.Initialize().IsOK());

    RunOptions run_options;
    run_options.run_tag = "one session/one tag";

    // prepare inputs
    std::vector<int64_t> dims_x = {3, 2};
    std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // prepare expected inputs and outputs
    std::vector<int64_t> expected_dims_y = {truncate_output ? 2 : 3, 2};
    // now the expected value should be Add's result.
    std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    if (truncate_output) {
      // The leading dimension is truncated, and the second dimension has two elements over that dim
      expected_values_y.resize(expected_values_y.size() - 2);
    }

    // Now run
    RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
  }
}

TEST(CustomKernelTests, CustomKernelWithOptionalOutput) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  //reigster optional schema
  auto optional_schema = GetOptionalOpSchema();
  std::vector<OpSchema> schemas = {optional_schema};

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  EXPECT_TRUE(registry->RegisterOpSet(schemas, LotusIR::kOnnxDomain, 5, 7).IsOK());
  auto def = OptionalKernelDef();
  //Register a foo kernel which is doing Add, but bind to Mul.
  EXPECT_TRUE(registry->RegisterCustomKernel(def, CreateOptionalOpKernel).IsOK());

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
  EXPECT_TRUE(session_object.Load(OPTIONAL_MODEL1_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be equal result.
  std::vector<float> expected_values_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}
}  // namespace Test
}  // namespace Lotus
