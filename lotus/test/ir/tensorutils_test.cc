#include "gtest/gtest.h"
#include "core/graph/tensorutils.h"
#include "core/protobuf/onnx-ml.pb.h"

using namespace Lotus::Utils;
using namespace onnx;

namespace Lotus
{
    namespace Test
    {
        TEST(TensorParseTest, TensorUtilsTest)
        {
            TensorProto boolTensorProto;
            boolTensorProto.set_data_type(TensorProto_DataType_BOOL);
            boolTensorProto.add_int32_data(1);

            bool boolData[1];
            auto status = TensorUtils::UnpackTensor(boolTensorProto, boolData, 1);
            EXPECT_TRUE(status.Ok());
            EXPECT_TRUE(boolData[0]);

            float floatData[1];
            status = TensorUtils::UnpackTensor(boolTensorProto, floatData, 1);
            EXPECT_FALSE(status.Ok());

            TensorProto floatTensorProto;
            floatTensorProto.set_data_type(TensorProto_DataType_FLOAT);
            float f[4] = { 1.1f, 2.2f, 3.3f, 4.4f };
            char rawdata[sizeof(float) * 4 + 1];
            for (int i = 0; i < 4; ++i)
            {
                memcpy(rawdata + i * sizeof(float), &(f[i]), sizeof(float));
            }

            rawdata[sizeof(float) * 4] = '\0';
            floatTensorProto.set_raw_data(rawdata);
            float floatData2[4];
            status = TensorUtils::UnpackTensor(floatTensorProto, floatData2, 4);
            EXPECT_TRUE(status.Ok());
            EXPECT_EQ(1.1f, floatData2[0]);
            EXPECT_EQ(2.2f, floatData2[1]);
            EXPECT_EQ(3.3f, floatData2[2]);
            EXPECT_EQ(4.4f, floatData2[3]);

            TensorProto stringTensorProto;
            stringTensorProto.set_data_type(TensorProto_DataType_STRING);
            stringTensorProto.add_string_data("a");
            stringTensorProto.add_string_data("b");

            std::string stringData[2];
            status = TensorUtils::UnpackTensor(stringTensorProto, stringData, 2);
            EXPECT_TRUE(status.Ok());
            EXPECT_EQ("a", stringData[0]);
            EXPECT_EQ("b", stringData[1]);

            status = TensorUtils::UnpackTensor(boolTensorProto, stringData, 2);
            EXPECT_FALSE(status.Ok());
        }
    }
}