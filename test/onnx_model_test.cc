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
            std::wstring filePath = L".\\testdata\\super_resolution.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
        TEST(ONNXModelsTest, bvlc_alexnet)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\bvlc_alexnet\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(DISABLED_ONNXModelsTest, densenet121)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\densenet121\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v1)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\inception_v1\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v2)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\inception_v2\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, resnet50)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\resnet50\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, shufflenet)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\shufflenet\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, squeezenet)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\squeezenet\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg16)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\vgg16\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg19)
        {
            ModelProto modelProto;
            std::wstring filePath = L".\\models\\vgg19\\model.pb";
            EXPECT_TRUE(Model::Load(filePath, &modelProto));
            Model model(modelProto);
            auto status = model.MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }
#endif
    }
}
