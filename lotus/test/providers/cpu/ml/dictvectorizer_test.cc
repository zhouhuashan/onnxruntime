#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MLOpTest, DictVectorizer) {
  OpTester test("DictVectorizer", LotusIR::kMLDomain);

  test.AddAttribute("string_vocabulary", std::vector<std::string>{"a", "b", "c", "d"});

  std::map<std::string, int64_t> map;
  map["a"] = 1;
  map["c"] = 2;
  map["d"] = 3;

  test.AddInput<std::string, int64_t>("X", map);

  std::vector<int64_t> dims{1, 4};
  test.AddOutput<int64_t>("Y", dims,
                          {1, 0, 2, 3});
  test.Run();
}

}  // namespace Test
}  // namespace Lotus
