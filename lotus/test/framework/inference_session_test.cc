#include "core/framework/inference_session.h"

#include <thread>
#include <functional>
#include "gtest/gtest.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"

namespace Lotus {
namespace Test {

TEST(InferenceSessionTestNoTimeout, RunTest) {
  SessionOptions so;
  so.num_threads = 1;
  InferenceSession is {so};
  std::vector<MLValue> feeds;
  std::vector<MLValue> fetches;
  RunOptions run_options;
  run_options.timeout_in_ms = 0;
  run_options.run_tag = "one session/one thread";
  Common::Status st = is.Run(run_options, feeds, &fetches);
  std::cout << "without timeout run status: " << st.ToString() << std::endl;
  EXPECT_TRUE(st.IsOK());
}

TEST(MultipleInferenceSessionTestNoTimeout, RunTest) {
  SessionOptions session_options;
  session_options.num_threads = 5;
  InferenceSession session_object {session_options};
  Common::Status st1;
  Common::Status st2;
  
  std::thread thread1 {[&session_object](Common::Status& st1) {
      std::vector<MLValue> feeds;
      std::vector<MLValue> fetches;
      RunOptions run_options;
      run_options.timeout_in_ms = 0;
      run_options.run_tag = "one session/multiple threads - first thread";      
      st1 = session_object.Run(run_options, feeds, &fetches);
    }, std::ref(st1)};

  std::thread thread2 {[&session_object](Common::Status& st2) {
      std::vector<MLValue> feeds;
      std::vector<MLValue> fetches;
      RunOptions run_options;
      run_options.timeout_in_ms = 0;
      run_options.run_tag = "one session/multiple threads - second thread";
      st2 = session_object.Run(run_options, feeds, &fetches);
    }, std::ref(st2)};

  thread1.join();
  thread2.join();

  EXPECT_TRUE(st1.IsOK() && st2.IsOK());
} 

// TODO write test with timeout

}
}
