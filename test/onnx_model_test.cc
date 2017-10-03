#include "gtest/gtest.h"
#include "graph.h"
#include "op.h"

namespace LotusIR
{
    namespace Test
    {
        TEST(ONNXModelsTest, super_resolution)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\super_resolution.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
        TEST(ONNXModelsTest, bvlc_alexnet)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\bvlc_alexnet\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(DISABLED_ONNXModelsTest, densenet121)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\densenet121\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v1)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\inception_v1\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v2)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\inception_v2\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, resnet50)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\resnet50\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, shufflenet)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\shufflenet\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, squeezenet)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\squeezenet\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg16)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\vgg16\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg19)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\models\\vgg19\\graph.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }
#endif
    }
}