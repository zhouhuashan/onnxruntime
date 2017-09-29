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

#ifdef false
        TEST(ONNXModelsTest, bvlc_alexnet)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\bvlc_alexnet.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, densenet121)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\densenet121.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v1)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\inception_v1.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, inception_v2)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\inception_v2.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, resnet50)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\resnet50.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, shufflenet)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\shufflenet.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, squeezenet)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\squeezenet.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg16)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\vgg16.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(ONNXModelsTest, vgg19)
        {
            GraphProto graphProto;
            std::wstring filePath = L".\\testdata\\vgg19.pb";
            EXPECT_TRUE(Graph::Load(filePath, &graphProto));
            Graph graph(graphProto);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }
#endif
    }
}