#include "core/framework/inference_session.h"

#include <thread>
#include <functional>
#include "gtest/gtest.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace Lotus {
REGISTER_PROVIDER(CPUExecutionProvider);
namespace Test {

static const std::string MODEL_URI = "testdata/super_resolution.pb";

void SetupFeedsAndOutputNames(const std::string& model_uri,
                              NameMLValMap& feeds,
                              std::vector<std::string>& output_names) {
  using namespace LotusIR;
  std::shared_ptr<Model> p_model;
  Model::Load(model_uri, &p_model);
  Graph* p_graph = p_model->MainGraph();
  const std::vector<const NodeArg*>& inputs = p_graph->GetInputs();
  const std::vector<const NodeArg*>& outputs = p_graph->GetOutputs();
  for (auto& elem : inputs) {
    feeds.insert(std::make_pair(elem->Name(), MLValue()));
  }
  for (auto& elem : outputs) {
    output_names.push_back(elem->Name());
  }
}

TEST(InferenceSessionTestNoTimeout, RunTest) {
  ExecutionProviderInfo epi;
  ProviderOption po {"CPUExecutionProvider", epi};
  SessionOptions so(vector<ProviderOption>{po}, true);
  InferenceSession session_object {so};
  EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());
  MLValue mlval;
  NameMLValMap feeds;
  std::vector<std::string> output_names;
  SetupFeedsAndOutputNames(MODEL_URI, feeds, output_names);
  std::vector<MLValue> fetches;
  RunOptions run_options;
  //run_options.timeout_in_ms = 0;
  run_options.run_tag = "one session/one thread";
  Common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  std::cout << "without timeout run status: " << st.ToString() << std::endl;
  EXPECT_TRUE(st.IsOK());
}

TEST(MultipleInferenceSessionTestNoTimeout, RunTest) {
  ExecutionProviderInfo epi;
  ProviderOption po {"CPUExecutionProvider", epi};
  SessionOptions session_options(vector<ProviderOption>{po}, true);
  session_options.ep_options.push_back(po);  
  InferenceSession session_object {session_options};
  EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());  
  Common::Status st1;
  Common::Status st2;
  
  std::thread thread1 {[&session_object](Common::Status& st1) {
      MLValue mlval;
      NameMLValMap feeds;
      std::vector<std::string> output_names;
      SetupFeedsAndOutputNames(MODEL_URI, feeds, output_names);      
      std::vector<MLValue> fetches;
      RunOptions run_options;
      //run_options.timeout_in_ms = 0;
      run_options.run_tag = "one session/multiple threads - first thread";      
      st1 = session_object.Run(run_options, feeds, output_names, &fetches);
    }, std::ref(st1)};

  std::thread thread2 {[&session_object](Common::Status& st2) {
      MLValue mlval;
      NameMLValMap feeds;
      std::vector<std::string> output_names;
      SetupFeedsAndOutputNames(MODEL_URI, feeds, output_names);      
      std::vector<MLValue> fetches;      
      RunOptions run_options;
      //run_options.timeout_in_ms = 0;
      run_options.run_tag = "one session/multiple threads - second thread";
      st2 = session_object.Run(run_options, feeds, output_names, &fetches);
    }, std::ref(st2)};

  thread1.join();
  thread2.join();

  EXPECT_TRUE(st1.IsOK() && st2.IsOK());
} 

// TODO write test with timeout

}
}
