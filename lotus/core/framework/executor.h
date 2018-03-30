#ifndef LOTUS_CORE_FRAMEWORK_EXECUTOR_H_
#define LOTUS_CORE_FRAMEWORK_EXECUTOR_H_

#include <vector>
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/execution_frame.h"
#include "core/framework/inference_session.h"
#include "core/framework/ml_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"

namespace Lotus {
class Executor {
 public:
  virtual ~Executor() {}

  static std::unique_ptr<Executor> NewSequentialExecutor(const SessionState& session_state,
                                                         const NameMLValMap& feeds, /* required for execution frame construction */
                                                         const std::vector<std::string>& output_names /* required for execution frame construction */);

  virtual Common::Status Execute(const RunOptions& run_options,
                                 const Logging::Logger& run_logger,
                                 const NameMLValMap& feeds,
                                 const std::vector<std::string>& output_names,
                                 std::vector<MLValue>* p_fetches);
};
}  // namespace Lotus

#endif  // LOTUS_CORE_FRAMEWORK_EXECUTOR_H_
