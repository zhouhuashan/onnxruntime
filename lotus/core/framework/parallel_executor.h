#pragma once

#include <vector>
#include <condition_variable>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"

namespace Lotus {

class ExecutionFrame;

class ParallelExecutor : public IExecutor {
 public:
  ParallelExecutor() = default;

  Common::Status Execute(const SessionState& session_state,
                         const NameMLValMap& feeds,
                         const std::vector<std::string>& output_names,
                         std::vector<MLValue>& fetches,
                         const Logging::Logger& logger) override;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(ParallelExecutor);

  void RunNodeAsync(size_t p_node_index, const SessionState& session_state, const Logging::Logger& logger);

  void EnqueueNode(size_t p_node_index, const SessionState& session_state, const Logging::Logger& logger);

  Status FetchOutput(const MLValueNameIdxMap& name_idx_map,
                     ExecutionFrame& frame,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>& fetches,
                     const Logging::Logger& logger);

  std::unique_ptr<ExecutionFrame> root_frame_;
  std::vector<size_t> node_refs_;
  std::mutex ref_mutex_;
  std::atomic<int> out_standings_;
  std::mutex complete_mutex_;
  std::condition_variable complete_cv_;
};
}  // namespace Lotus
