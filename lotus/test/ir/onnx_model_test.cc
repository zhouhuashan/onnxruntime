#include <memory>
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "gtest/gtest.h"

namespace LotusIR
{
    namespace Test
    {
        // Tests that Resolve() properly clears the state of topological sorted nodes,
        // inputs, outputs and valueInfo.
        // Assumes the graph passed in has been previously resolved.
        void TestResolve(LotusIR::Graph* p_graph)
        {
            std::vector<LotusIR::NODEINDEX>* nodes;
            p_graph->GetNodesInTopologicalOrder(&nodes);
            auto nodesBefore = *nodes;
            auto inputsBefore = p_graph->GetInputs();
            auto outputsBefore = p_graph->GetOutputs();
            auto valueInfoBefore = p_graph->GetValueInfo();

            // Touch the graph to force Resolve() to recompute.
            p_graph->GetNode(0)->Mutable_InputArgCount();
            EXPECT_TRUE(p_graph->Resolve().IsOK());

            std::vector<LotusIR::NODEINDEX>* nodesAfter;
            p_graph->GetNodesInTopologicalOrder(&nodesAfter);
            auto& inputsAfter = p_graph->GetInputs();
            auto& outputsAfter = p_graph->GetOutputs();
            auto& valueInfoAfter = p_graph->GetValueInfo();

            // Multiple calls to Resolve() should not alter the sorted nodes,
            // inputs, outputs and valueInfo. The internal state should be
            // cleared.
            EXPECT_EQ(nodesBefore, *nodesAfter);
            EXPECT_EQ(inputsBefore, inputsAfter);
            EXPECT_EQ(outputsBefore, outputsAfter);
            EXPECT_EQ(valueInfoBefore, valueInfoAfter);
        }

        TEST(ONNXModelsTest, super_resolution)
        {
            // NOTE: this requires the current directory to be where LotusIR_UT.exe is located
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./testdata/super_resolution.pb", &model).IsOK());
            TestResolve(model->MainGraph());
#ifdef _WIN32
            // wstring version
            std::shared_ptr<Model> model2;
            EXPECT_TRUE(Model::Load(L"./testdata/super_resolution.pb", &model2).IsOK());
            TestResolve(model2->MainGraph());
#endif
        }

#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
        TEST(ONNXModelsTest, bvlc_alexnet)
        {
            using ::google::protobuf::io::ZeroCopyInputStream;
            using ::google::protobuf::io::FileInputStream;
            using ::google::protobuf::io::CodedInputStream;

            int fd;
            FileOpenRd("./models/bvlc_alexnet/model.pb", &fd);
            std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
            std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));
            // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
            coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);
            ModelProto modelProto;
            bool result = modelProto.ParseFromCodedStream(coded_input.get());
            coded_input.reset();
            raw_input.reset();
            EXPECT_TRUE(result);
            FileClose(fd);

            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/bvlc_alexnet/model.pb", &model).IsOK());

            // Check the graph input/output/value_info should have the same size as specified in the model file.
            EXPECT_EQ(modelProto.graph().value_info_size(), model->MainGraph()->GetValueInfo().size());
            EXPECT_EQ(modelProto.graph().input_size(), model->MainGraph()->GetInputs().size());
            EXPECT_EQ(modelProto.graph().output_size(), model->MainGraph()->GetOutputs().size());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, densenet121)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/densenet121/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, inception_v1)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/inception_v1/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, inception_v2)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/inception_v2/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, resnet50)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/resnet50/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, shufflenet)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/shufflenet/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, squeezenet)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/squeezenet/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, vgg16)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/vgg16/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }

        TEST(ONNXModelsTest, vgg19)
        {
            std::shared_ptr<Model> model;
            EXPECT_TRUE(Model::Load("./models/vgg19/model.pb", &model).IsOK());
            TestResolve(model->MainGraph());
        }
#endif
    }
}
