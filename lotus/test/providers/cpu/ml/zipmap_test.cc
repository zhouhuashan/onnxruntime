#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

template <typename T>
void TestHelper(const std::vector<T>& classes, const std::string& type) {
  OpTester test("ZipMap", LotusIR::kMLDomain);

  std::vector<float> input{1.f, 0.f, 3.f, 44.f, 23.f, 11.3f};
  vector<int64_t> dims{2, 3};

  if (type == "string") {
    test.AddAttribute("classlabels_strings", classes);
  } else if (type == "int64_t") {
    test.AddAttribute("classlabels_int64s", classes);
  } else {
    LOTUS_THROW("Invalid type: ", type);
  }

  // prepare expected output
  std::vector<std::map<T, float>> expected_output;
  for (int64_t i = 0; i < dims[0]; ++i) {
    std::map<T, float> var_map;
    for (size_t j = 0; j < classes.size(); ++j) {
      var_map.emplace(classes[j], input[i * 3 + j]);
    }
    expected_output.push_back(var_map);
  }

  test.AddInput<float>("X", dims, input);
  test.AddOutput<T, float>("Z", expected_output);
  test.Run();
}

TEST(MLOpTest, ZipMapOpStringFloat) {
  TestHelper<string>({"class1", "class2", "class3"}, "string");
}

TEST(MLOpTest, ZipMapOpInt64Float) {
  TestHelper<int64_t>({10, 20, 30}, "int64_t");
}

}  // namespace Test
}  // namespace Lotus
