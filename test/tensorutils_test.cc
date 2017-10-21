#include "gtest/gtest.h"
#include "core/graph/tensorutils.h"
#include "core/protobuf/graph.pb.h"

using namespace LotusIR;
using namespace Lotus::Utils;

namespace Lotus
{
    namespace Test
    {
        TEST(TensorParseTest, TensorUtilsTest)
        {
            TensorProto boolTensorProto;
            boolTensorProto.set_data_type(TensorProto_DataType_BOOL);
            boolTensorProto.add_int32_data(1);

            std::vector<bool> boolData;
            auto status = TensorUtils::UnpackTensor(boolTensorProto, &boolData);
            EXPECT_TRUE(status.Ok());
            EXPECT_EQ(1, boolData.size());
            EXPECT_TRUE(boolData[0]);

            std::vector<float> floatData;
            status = TensorUtils::UnpackTensor(boolTensorProto, &floatData);
            EXPECT_FALSE(status.Ok());

            TensorProto floatTensorProto;
            floatTensorProto.set_data_type(TensorProto_DataType_FLOAT);
            char rawdata[sizeof(float)+1];
            float f = 1.1f;
            memcpy(rawdata, &f, sizeof f);
            rawdata[sizeof(float)] = '\0';
            floatTensorProto.set_raw_data(rawdata);
            TensorUtils::UnpackTensor(floatTensorProto, &floatData);
            EXPECT_EQ(1, floatData.size());
            EXPECT_EQ(1.1f, floatData[0]);

            TensorProto stringTensorProto;
            stringTensorProto.set_data_type(TensorProto_DataType_STRING);
            stringTensorProto.add_string_data("a");
            stringTensorProto.add_string_data("b");

            std::vector<std::string> stringData;
            status = TensorUtils::UnpackTensor(stringTensorProto, &stringData);
            EXPECT_TRUE(status.Ok());
            EXPECT_EQ(2, stringData.size());
            EXPECT_EQ("a", stringData[0]);
            EXPECT_EQ("b", stringData[1]);

            status = TensorUtils::UnpackTensor(boolTensorProto, &stringData);
            EXPECT_FALSE(status.Ok());
        }
    }
}