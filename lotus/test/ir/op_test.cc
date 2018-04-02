#include "core/graph/op.h"
#include <iostream>
#include "core/graph/utils.h"
#include "onnx/onnx-ml.pb.h"
#include "gtest/gtest.h"

using namespace onnx;

namespace LotusIR {
namespace Test {
TEST(FormalParamTest, Success) {
  OpSchema::FormalParameter p("input", "desc: integer input", "tensor(int32)");
  EXPECT_EQ("input", p.GetName());
  EXPECT_EQ("tensor(int32)", p.GetTypeStr());
  EXPECT_EQ("desc: integer input", p.GetDescription());
  // TODO: change onnx to make formal parameter construction self-contain.
  //EXPECT_EQ(Utils::DataTypeUtils::ToType("tensor(int32)"), *p.GetTypes().begin());
}

TEST(OpRegistrationTest, OpRegTest) {
  ONNX_OPERATOR_SCHEMA(__TestOpReg)
      .SetDoc("Op Registration Basic Test.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  const OpSchema* opSchema = OpSchemaRegistry::Schema("__TestOpReg");
  EXPECT_TRUE(nullptr != opSchema);
  EXPECT_EQ(opSchema->inputs().size(), 2);
  EXPECT_EQ(opSchema->inputs()[0].GetName(), "input_1");
  EXPECT_EQ(opSchema->inputs()[0].GetTypes().size(), 1);
  EXPECT_EQ(**opSchema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(int32)")), "tensor(int32)");
  EXPECT_EQ(opSchema->inputs()[1].GetName(), "input_2");
  EXPECT_EQ(opSchema->inputs()[1].GetTypes().size(), 1);
  EXPECT_EQ(**opSchema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(int32)")), "tensor(int32)");
  EXPECT_EQ(opSchema->outputs().size(), 1);
  EXPECT_EQ(opSchema->outputs()[0].GetName(), "output_1");
  EXPECT_EQ(opSchema->outputs()[0].GetTypes().size(), 1);
  EXPECT_EQ(**opSchema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(int32)")), "tensor(int32)");
}

TEST(OpRegistrationTest, TypeConstraintTest) {
  ONNX_OPERATOR_SCHEMA(__TestTypeConstraint)
      .SetDoc("Op with Type Constraint.")
      .Input(0, "input_1", "docstr for input_1.", "T")
      .Input(1, "input_2", "docstr for input_2.", "T")
      .Output(0, "output_1", "docstr for output_1.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to floats.");
  const OpSchema* opSchema = OpSchemaRegistry::Schema("__TestTypeConstraint");
  EXPECT_TRUE(nullptr != opSchema);
  EXPECT_EQ(opSchema->inputs().size(), 2);
  EXPECT_EQ(opSchema->inputs()[0].GetName(), "input_1");
  EXPECT_EQ(opSchema->inputs()[0].GetTypes().size(), 3);
  EXPECT_EQ(**opSchema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float16)")), "tensor(float16)");
  EXPECT_EQ(**opSchema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float)")), "tensor(float)");
  EXPECT_EQ(**opSchema->inputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(double)")), "tensor(double)");

  EXPECT_EQ(opSchema->inputs()[1].GetName(), "input_2");
  EXPECT_EQ(opSchema->inputs()[1].GetTypes().size(), 3);
  EXPECT_EQ(**opSchema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float16)")), "tensor(float16)");
  EXPECT_EQ(**opSchema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float)")), "tensor(float)");
  EXPECT_EQ(**opSchema->inputs()[1].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(double)")), "tensor(double)");

  EXPECT_EQ(opSchema->outputs().size(), 1);
  EXPECT_EQ(opSchema->outputs()[0].GetName(), "output_1");
  EXPECT_EQ(opSchema->outputs()[0].GetTypes().size(), 3);
  EXPECT_EQ(**opSchema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float16)")), "tensor(float16)");
  EXPECT_EQ(**opSchema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(float)")), "tensor(float)");
  EXPECT_EQ(**opSchema->outputs()[0].GetTypes().find(Utils::DataTypeUtils::ToType("tensor(double)")), "tensor(double)");
}

TEST(OpRegistrationTest, AttributeDefaultValueTest) {
  ONNX_OPERATOR_SCHEMA(__TestAttrDefaultValue)
      .SetDoc("Op with attributes that have default values")
      .Attr("my_attr_int", "attr with default value of 99.", AttrType::AttributeProto_AttributeType_INT, int64_t(99))
      .Attr("my_attr_float", "attr with default value of 0.99.", AttrType::AttributeProto_AttributeType_FLOAT, float(0.99))
      .Attr("my_attr_string", "attr with default value of \"99\".", AttrType::AttributeProto_AttributeType_STRING, std::string("99"));
  const OpSchema* opSchema = OpSchemaRegistry::Schema("__TestAttrDefaultValue");
  EXPECT_TRUE(nullptr != opSchema);
  EXPECT_EQ(opSchema->attributes().size(), 3);

  auto attr_int = opSchema->attributes().find("my_attr_int")->second;
  EXPECT_EQ(attr_int.name, "my_attr_int");
  EXPECT_EQ(attr_int.type, AttrType::AttributeProto_AttributeType_INT);
  EXPECT_FALSE(attr_int.required);
  EXPECT_EQ(attr_int.default_value.name(), "my_attr_int");
  EXPECT_TRUE(attr_int.default_value.has_i());
  EXPECT_EQ(attr_int.default_value.i(), 99LL);

  auto attr_float = opSchema->attributes().find("my_attr_float")->second;
  EXPECT_EQ(attr_float.name, "my_attr_float");
  EXPECT_EQ(attr_float.type, AttrType::AttributeProto_AttributeType_FLOAT);
  EXPECT_FALSE(attr_float.required);
  EXPECT_EQ(attr_float.default_value.name(), "my_attr_float");
  EXPECT_TRUE(attr_float.default_value.has_f());
  EXPECT_EQ(attr_float.default_value.f(), 0.99f);

  auto attr_string= opSchema->attributes().find("my_attr_string")->second;
  EXPECT_EQ(attr_string.name, "my_attr_string");
  EXPECT_EQ(attr_string.type, AttrType::AttributeProto_AttributeType_STRING);
  EXPECT_FALSE(attr_string.required);
  EXPECT_EQ(attr_string.default_value.name(), "my_attr_string");
  EXPECT_TRUE(attr_string.default_value.has_s());
  EXPECT_EQ(attr_string.default_value.s(), "99");
}

TEST(OpRegistrationTest, AttributeDefaultValueListTest) {
  ONNX_OPERATOR_SCHEMA(__TestAttrDefaultValueList)
      .SetDoc("Op with attributes that have default list of values.")
      .Attr("my_attr_ints", "attr with default value of [98, 99, 100].", AttrType::AttributeProto_AttributeType_INTS, std::vector<int64_t>{int64_t(98), int64_t(99), int64_t(100)}).Attr("my_attr_floats", "attr with default value of [0.98, 0.99, 1.00].", AttrType::AttributeProto_AttributeType_FLOATS, std::vector<float>{float(0.98), float(0.99), float(1.00)}).Attr("my_attr_strings", "attr with default value of [\"98\", \"99\", \"100\"].", AttrType::AttributeProto_AttributeType_STRINGS, std::vector<std::string>{"98", "99", "100"});
  const OpSchema* opSchema = OpSchemaRegistry::Schema("__TestAttrDefaultValueList");
  EXPECT_TRUE(nullptr != opSchema);
  EXPECT_EQ(opSchema->attributes().size(), 3);

  auto attr_ints = opSchema->attributes().find("my_attr_ints")->second;
  EXPECT_EQ(attr_ints.name, "my_attr_ints");
  EXPECT_EQ(attr_ints.type, AttrType::AttributeProto_AttributeType_INTS);
  EXPECT_FALSE(attr_ints.required);
  EXPECT_EQ(attr_ints.default_value.name(), "my_attr_ints");
  int size = attr_ints.default_value.ints_size();
  EXPECT_EQ(size, 3);
  std::vector<int64_t> expected_ints = {98LL, 99LL, 100LL};
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(attr_ints.default_value.ints(i), expected_ints[i]);
  }

  auto attr = opSchema->attributes().find("my_attr_floats")->second;
  EXPECT_EQ(attr.name, "my_attr_floats");
  EXPECT_EQ(attr.type, AttrType::AttributeProto_AttributeType_FLOATS);
  EXPECT_FALSE(attr.required);
  EXPECT_EQ(attr.default_value.name(), "my_attr_floats");
  size = attr.default_value.floats_size();
  EXPECT_EQ(size, 3);
  std::vector<float> expected_floats = {0.98f, 0.99f, 1.00f};
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(attr.default_value.floats(i), expected_floats[i]);
  }

  auto attr2 = opSchema->attributes().find("my_attr_strings")->second;
  EXPECT_EQ(attr2.name, "my_attr_strings");
  EXPECT_EQ(attr2.type, AttrType::AttributeProto_AttributeType_STRINGS);
  EXPECT_FALSE(attr2.required);
  EXPECT_EQ(attr2.default_value.name(), "my_attr_strings");
  size = attr2.default_value.strings_size();
  EXPECT_EQ(size, 3);
  std::vector<std::string> expected_strings = {"98", "99", "100"};
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(attr2.default_value.strings(i), expected_strings[i]);
  }
}

}  // namespace Test
}  // namespace LotusIR
