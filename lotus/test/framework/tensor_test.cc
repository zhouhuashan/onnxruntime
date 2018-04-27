#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
template <typename T>
void CPUTensorTest(std::vector<int64_t> dims, const int offset = 0) {
  //add scope to explicitly delete tensor
  {
    //not own the buffer
    TensorShape shape(dims);
    auto& alloc = AllocatorManager::Instance().GetArena(CPU);
    auto data = alloc.Alloc(sizeof(T) * (shape.Size() + offset));
    EXPECT_TRUE(data);
    Tensor t(DataTypeImpl::GetType<T>(), shape, data, alloc.Info(), nullptr, offset);
    auto tensor_shape = t.Shape();
    EXPECT_EQ(shape, tensor_shape);
    EXPECT_EQ(t.DataType(), DataTypeImpl::GetType<T>());
    auto& location = t.Location();
    EXPECT_EQ(location.name, CPU);
    EXPECT_EQ(location.id, 0);

    auto t_data = t.MutableData<T>();
    EXPECT_TRUE(t_data);
    memset(t_data, 0, sizeof(T) * shape.Size());
    EXPECT_EQ(*(T*)((char*)data + offset), (T)0);
    alloc.Free(data);

    // owned buffer
    data = alloc.Alloc(sizeof(T) * (shape.Size() + offset));
    EXPECT_TRUE(data);
    Tensor new_t(DataTypeImpl::GetType<T>(), shape, data, alloc.Info(), &alloc, offset);

    tensor_shape = new_t.Shape();
    EXPECT_EQ(shape, tensor_shape);
    EXPECT_EQ(new_t.DataType(), DataTypeImpl::GetType<T>());
    auto& new_location = new_t.Location();
    EXPECT_EQ(new_location.name, CPU);
    EXPECT_EQ(new_location.id, 0);

    auto new_data = new_t.MutableData<T>();
    EXPECT_TRUE(new_data);
    memset(new_data, 0, sizeof(T) * shape.Size());
    EXPECT_EQ(*(T*)((char*)new_data + offset), (T)0);
    //no free op as the tensor own the buffer
  }
}

TEST(TensorTest, CPUFloatTensorTest) {
  CPUTensorTest<float>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUInt32TensorTest) {
  CPUTensorTest<int32_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt8TensorTest) {
  CPUTensorTest<uint8_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt16TensorTest) {
  CPUTensorTest<uint16_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUInt16TensorTest) {
  CPUTensorTest<int16_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUInt64TensorTest) {
  CPUTensorTest<int64_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUDoubleTensorTest) {
  CPUTensorTest<double>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt32TensorTest) {
  CPUTensorTest<uint32_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt64TensorTest) {
  CPUTensorTest<uint64_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUFloatTensorOffsetTest) {
  CPUTensorTest<float>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUInt32TensorOffsetTest) {
  CPUTensorTest<int32_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt8TensorOffsetTest) {
  CPUTensorTest<uint8_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt16TensorOffsetTest) {
  CPUTensorTest<uint16_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUInt16TensorOffsetTest) {
  CPUTensorTest<int16_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUInt64TensorOffsetTest) {
  CPUTensorTest<int64_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUDoubleTensorOffsetTest) {
  CPUTensorTest<double>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt32TensorOffsetTest) {
  CPUTensorTest<uint32_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt64TensorOffsetTest) {
  CPUTensorTest<uint64_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, EmptyTensorTest) {
  Tensor t;
  auto& shape = t.Shape();
  EXPECT_EQ(shape.Size(), 0);
  EXPECT_EQ(t.DataType(), DataTypeImpl::GetType<float>());

  auto data = t.MutableData<float>();
  EXPECT_TRUE(!data);

  auto& location = t.Location();
  EXPECT_EQ(location.name, CPU);
  EXPECT_EQ(location.id, 0);
  EXPECT_EQ(location.type, AllocatorType::kArenaAllocator);
}

TEST(TensorTest, TensorCopyAssignOpTest) {
  TensorShape shape({1, 2, 3});
  auto& alloc = AllocatorManager::Instance().GetArena(CPU);
  auto data = alloc.Alloc(sizeof(int) * shape.Size());
  EXPECT_TRUE(data);
  Tensor t1(DataTypeImpl::GetType<int>(), shape, data, alloc.Info());
  Tensor t2 = t1;
  EXPECT_EQ(t2.DataType(), DataTypeImpl::GetType<int>());
  EXPECT_EQ(t2.Shape(), shape);
  auto location = t2.Location();
  EXPECT_EQ(location.name, CPU);
  EXPECT_EQ(location.id, 0);
  EXPECT_EQ(location.type, AllocatorType::kArenaAllocator);
  auto t_data = t2.Data<int>();
  EXPECT_EQ((void*)t_data, data);
  alloc.Free(data);
}

TEST(TensorTest, StringTensorTest) {
  //add scope to explicitly delete tensor
  std::string* string_ptr = nullptr;
  {
    TensorShape shape({2, 3});
    auto& alloc = AllocatorManager::Instance().GetArena(CPU);
    auto buffer = alloc.Alloc(sizeof(std::string) * (shape.Size()));
    Tensor t(DataTypeImpl::GetType<std::string>(), shape, buffer, alloc.Info(), &alloc);

    auto& tensor_shape = t.Shape();
    EXPECT_EQ(shape, tensor_shape);
    EXPECT_EQ(t.DataType(), DataTypeImpl::GetType<std::string>());
    auto& location = t.Location();
    EXPECT_EQ(location.name, CPU);
    EXPECT_EQ(location.id, 0);

    std::string* new_data = t.MutableData<std::string>();
    EXPECT_TRUE(new_data);
    new_data[0] = "a";
    new_data[1] = "b";

    auto tensor_data = t.Data<std::string>();
    EXPECT_EQ(tensor_data[0], "a");
    EXPECT_EQ(tensor_data[1], "b");
    string_ptr = new_data;
  }
  // on msvc, check does the ~string be called when release tensor
  // It may be not stable as access to a deleted pointer could have
  // undefined behavior. If we find it is failure on other platform
  // go ahead to remove it.
#ifdef _MSC_VER
  EXPECT_EQ(string_ptr->size(), 0);
  EXPECT_EQ((string_ptr + 1)->size(), 0);
#endif
}
}  // namespace Test
}  // namespace Lotus
