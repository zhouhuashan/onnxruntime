#include "gtest/gtest.h"
#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "gtest/gtest.h"


namespace Lotus
{
    namespace Test
    {
        template<typename T>
        void CPUTensorTest(std::vector<int64_t> dims, const int offset = 0)
        {
            TensorShape shape(dims);
            auto& alloc = AllocatorManager::Instance()->GetArena(CPU);
            auto data = alloc.Alloc(sizeof(T) * (shape.Size() + offset));
            EXPECT_TRUE(data);
            Tensor t(DataTypeImpl::GetTensorType<T>(), shape, data, alloc.Info(), offset);
            auto tensor_shape = t.shape();
            EXPECT_EQ(shape, tensor_shape);
            EXPECT_EQ(t.dtype(), DataTypeImpl::GetTensorType<T>());
            auto& location = t.location();
            EXPECT_EQ(location.name_, CPU);
            EXPECT_EQ(location.id_, 0);

            auto t_data = t.mutable_data<T>();
            EXPECT_TRUE(t_data);
            memset(t_data, 0, sizeof(T) * shape.Size());
            EXPECT_EQ(*(T*)((char*)data + offset), (T)0);
            alloc.Free(data, sizeof(T) * shape.Size());
        }

        TEST(TensorTest, CPUFloatTensorTest)
        {
            CPUTensorTest<float>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUInt32TensorTest)
        {
            CPUTensorTest<int>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUUInt8TensorTest)
        {
            CPUTensorTest<uint8_t>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUUInt16TensorTest)
        {
            CPUTensorTest<uint16_t>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUInt16TensorTest)
        {
            CPUTensorTest<int16_t>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUInt64TensorTest)
        {
            CPUTensorTest<int64_t>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUDoubleTensorTest)
        {
            CPUTensorTest<double>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUUInt32TensorTest)
        {
            CPUTensorTest<uint32_t>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUUInt64TensorTest)
        {
            CPUTensorTest<uint64_t>(std::vector<int64_t>({ 3, 2, 4 }));
        }

        TEST(TensorTest, CPUFloatTensorOffsetTest)
        {
            CPUTensorTest<float>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUInt32TensorOffsetTest)
        {
            CPUTensorTest<int>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUUInt8TensorOffsetTest)
        {
            CPUTensorTest<uint8_t>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUUInt16TensorOffsetTest)
        {
            CPUTensorTest<uint16_t>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUInt16TensorOffsetTest)
        {
            CPUTensorTest<int16_t>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUInt64TensorOffsetTest)
        {
            CPUTensorTest<int64_t>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUDoubleTensorOffsetTest)
        {
            CPUTensorTest<double>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUUInt32TensorOffsetTest)
        {
            CPUTensorTest<uint32_t>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }

        TEST(TensorTest, CPUUInt64TensorOffsetTest)
        {
            CPUTensorTest<uint64_t>(std::vector<int64_t>({ 3, 2, 4 }), 5);
        }
        
        TEST(TensorTest, EmptyTensorTest)
        {
            Tensor t;
            auto& shape = t.shape();
            EXPECT_EQ(shape.Size(), 0);
            EXPECT_EQ(t.dtype(), DataTypeImpl::GetTensorType<float>());

            auto data = t.mutable_data<float>();
            EXPECT_TRUE(!data);

            auto& location = t.location();
            EXPECT_EQ(location.name_, CPU);
            EXPECT_EQ(location.id_, 0);
            EXPECT_EQ(location.type_, ArenaAllocator);
        }

        TEST(TensorTest, TensorCopyAssignOpTest)
        {
            TensorShape shape({1, 2, 3});
            auto& alloc = AllocatorManager::Instance()->GetArena(CPU);
            auto data = alloc.Alloc(sizeof(int) * shape.Size());
            EXPECT_TRUE(data);
            Tensor t1(DataTypeImpl::GetTensorType<int>(), shape, data, alloc.Info());
            Tensor t2 = t1;
            EXPECT_EQ(t2.dtype(), DataTypeImpl::GetTensorType<int>());
            EXPECT_EQ(t2.shape(), shape);
            auto location = t2.location();
            EXPECT_EQ(location.name_, CPU);
            EXPECT_EQ(location.id_, 0);
            EXPECT_EQ(location.type_, AllocatorType::ArenaAllocator);
            auto t_data = t2.data<int>();
            EXPECT_EQ((void*)t_data, data);
            alloc.Free(data, sizeof(int) * shape.Size());
        }        
    }
}