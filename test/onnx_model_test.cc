#include "gtest/gtest.h"
#include "graph.h"
#include "model.h"
#include "op.h"

namespace LotusIR
{
    namespace Test
    {
        TEST(ONNXModelsTest, super_resolution)
        {
            // NOTE: this requires the current directory to be where LotusIR_UT.exe is located
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./testdata/super_resolution.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
        TEST(ONNXModelsTest, bvlc_alexnet)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/bvlc_alexnet/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(DISABLED_ONNXModelsTest, densenet121)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/densenet121/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v1)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/inception_v1/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v2)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/inception_v2/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, resnet50)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/resnet50/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, shufflenet)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/shufflenet/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, squeezenet)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/squeezenet/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg16)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/vgg16/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg19)
        {
            ModelProto modelProto;
            EXPECT_TRUE(Model::Load("./models/vgg19/model.pb", &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }
#endif
    }
}
