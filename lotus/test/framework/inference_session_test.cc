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

TEST(InferenceSessionTestNoTimeout, RunTest) {
  ExecutionProviderInfo epi;
  ProviderOption po {"CPUExecutionProvider", epi};
  SessionOptions so;
  so.ep_options.push_back(po);
  InferenceSession session_object {so};
  EXPECT_TRUE(session_object.Load("testdata/super_resolution.pb").IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());
  NameMLValMap feeds;
  std::vector<std::string> output_names;  
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
  SessionOptions session_options;
  session_options.ep_options.push_back(po);  
  InferenceSession session_object {session_options};
  EXPECT_TRUE(session_object.Load("testdata/super_resolution.pb").IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());  
  Common::Status st1;
  Common::Status st2;
  
  std::thread thread1 {[&session_object](Common::Status& st1) {
      NameMLValMap feeds;
      std::vector<std::string> output_names;
      std::vector<MLValue> fetches;
      RunOptions run_options;
      //run_options.timeout_in_ms = 0;
      run_options.run_tag = "one session/multiple threads - first thread";      
      st1 = session_object.Run(run_options, feeds, output_names, &fetches);
    }, std::ref(st1)};

  std::thread thread2 {[&session_object](Common::Status& st2) {
      NameMLValMap feeds;
      std::vector<MLValue> fetches;
      std::vector<std::string> output_names;      
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
