#include "core/session/inference_session.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/graph_transformer.h"
#include "core/graph/identity_elimination.h"

#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;

using namespace onnx;

namespace onnxruntime {
namespace Test {

static const std::string MODEL_FOLDER = "testdata/transform/";

TEST(GraphTransformationTests, IdentityElimination) {
  string model_uri = MODEL_FOLDER + "abs-id-max.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  //Graph& p_graph = p_model->MainGraph();

  std::unique_ptr<TopDownRuleBasedTransformer> rule_transformer =
      std::make_unique<TopDownRuleBasedTransformer>("RuleTransformer1", "First rule transformer");

  rule_transformer->Register("Identity", std::make_unique<EliminateIdentity>());

  session_object.RegisterGraphTransformer(std::move(rule_transformer));

  ASSERT_TRUE(session_object.Initialize().IsOK());
}

}  // namespace Test
}  // namespace onnxruntime
