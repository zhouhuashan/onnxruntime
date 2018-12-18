// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(DynamicSliceTest, dynamic_slice_D2_varied_types) {
  OpTester test1 ("DynamicSlice", 9);
  test1.AddInput  <int32_t> ("data",   {3,3}, {1,2,3,4,5,6,7,8,9});
  test1.AddInput  <int32_t> ("starts", {2},   {1,1});
  test1.AddInput  <int32_t> ("ends",   {2},   {3,3});
  test1.AddOutput <int32_t> ("output", {2,2}, {5,6,8,9});
  test1.Run();

  OpTester test2("DynamicSlice", 9);
  test2.AddInput  <int64_t> ("data",   {3,3}, {1LL,2LL,3LL,4LL,5LL,6LL,7LL,8LL,9LL});
  test2.AddInput  <int32_t> ("starts", {2},   {1,1});
  test2.AddInput  <int32_t> ("ends",   {2},   {3,3});
  test2.AddOutput <int64_t> ("output", {2,2}, {5LL,6LL,8LL,9LL});
  test2.Run();

  OpTester test3("DynamicSlice", 9);
  test3.AddInput  <std::string> ("data",   {3,3}, {"a","b","c","d","e","f","g","h","i"});
  test3.AddInput  <int64_t>     ("starts", {2},   {1,1});
  test3.AddInput  <int64_t>     ("ends",   {2},   {3,3});
  test3.AddOutput <std::string> ("output", {2,2}, {"e","f","h","i"});
  test3.Run();

  OpTester test4("DynamicSlice", 9);
  test4.AddInput  <float>    ("data",   {3,3}, {1.1f,2.2f,3.3f,4.4f,5.5f,6.6f,7.7f,8.8f,9.9f});
  test4.AddInput  <int32_t>  ("starts", {2},   {1,1});
  test4.AddInput  <int32_t>  ("ends",   {2},   {3,3});
  test4.AddOutput <float>    ("output", {2,2}, {5.5f,6.6f,8.8f,9.9f});
  test4.Run();

  OpTester test5("DynamicSlice", 9);
  test5.AddInput  <bool>    ("data",   {3,3}, {false,true,false,false,false,false,true,false,true});
  test5.AddInput  <int32_t> ("starts", {2},   {1,1});
  test5.AddInput  <int32_t> ("ends",   {2},   {3,3});
  test5.AddOutput <bool>    ("output", {2,2}, {false,false,false,true});
  test5.Run();
}

}  // namespace Test
}  // namespace onnxruntime
