#include "gtest/gtest.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"

namespace LotusIR
{
    namespace Test
    {
        TEST(ONNXModelsTest, super_resolution)
        {
            // NOTE: this requires the current directory to be where LotusIR_UT.exe is located
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./testdata/super_resolution.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
#ifdef _WIN32
            // wstring version
            std::shared_ptr<Model> model2;
            EXPECT_TRUE(Model::Load(L"./testdata/super_resolution.pb", &model2).Ok());
            status = model2->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
#endif
        }

#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
        TEST(ONNXModelsTest, bvlc_alexnet)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/bvlc_alexnet/model.pb", &model).Ok());

            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(DISABLED_ONNXModelsTest, densenet121)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/densenet121/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v1)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/inception_v1/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v2)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/inception_v2/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, resnet50)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/resnet50/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, shufflenet)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/shufflenet/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, squeezenet)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/squeezenet/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg16)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/vgg16/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg19)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/vgg19/model.pb", &model).Ok());
            auto status = model->MainGraph()->Resolve();
            EXPECT_TRUE(status.Ok());
        }
#endif
    }
}
