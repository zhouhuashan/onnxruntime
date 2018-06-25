#include "profiler.h"

namespace Lotus {
namespace Profiling {
using namespace std::chrono;

Lotus::TimePoint Profiling::Profiler::StartTime() const {
  return std::chrono::high_resolution_clock::now();
}

void Profiler::StartProfiling(const Logging::Logger* session_logger, const std::string& file_name) {
  LOTUS_ENFORCE(session_logger != nullptr);
  session_logger_ = session_logger;
  enabled_ = true;
  profile_stream_ = std::ofstream(file_name, std::ios::out | std::ios::trunc);
  profile_stream_file_ = file_name;
  profiling_start_time_ = StartTime();
}

void Profiler::EndTimeAndRecordEvent(EventCategory category,
                                     const std::string& event_name,
                                     TimePoint& start_time,
                                     bool /*sync_gpu*/) {
  if (!enabled_)
    return;
  //TODO: sync_gpu if needed.
  std::lock_guard<std::mutex> lock(mutex_);
  if (timing_events_.size() < max_num_events_) {
    long long dur = TimeDiffMicroSeconds(start_time);
    long long ts = TimeDiffMicroSeconds(profiling_start_time_, start_time);
    timing_events_.emplace_back(category, Logging::GetProcessId(),
                                Logging::GetThreadId(), event_name, ts, dur);
  } else {
    if (session_logger_ && !max_events_reached) {
      LOGS(*session_logger_, ERROR)
          << "Maximum number of events reached, could not record profile event.";
      max_events_reached = true;
    }
  }
}

std::string Profiler::WriteProfileData() {
  std::lock_guard<std::mutex> lock(mutex_);
  profile_stream_ << "[\n";

  for (int i = 0; i < timing_events_.size(); ++i) {
    auto& rec = timing_events_[i];
    profile_stream_ << "{\"cat\" : \"" << event_categor_names_[rec.cat] << "\",";
    profile_stream_ << "\"pid\" :" << rec.pid << ",";
    profile_stream_ << "\"tid\" :" << rec.tid << ",";
    profile_stream_ << "\"dur\" :" << rec.dur << ",";
    profile_stream_ << "\"ts\" :" << rec.ts << ",";
    profile_stream_ << "\"ph\" : \"X\",";
    profile_stream_ << "\"name\" :\"" << rec.name << "\",";
    profile_stream_ << "\"args\" : {}";
    if (i == timing_events_.size() - 1) {
      profile_stream_ << "}\n";
    } else {
      profile_stream_ << "},\n";
    }
  }
  profile_stream_ << "]\n";
  profile_stream_.close();
  enabled_ = false;  // will not collect profile after writing.
  return profile_stream_file_;
}

//
// Conditionally sync the GPU if the syncGPU flag is set.
//
void ProfilerSyncGpu() {
  LOTUS_NOT_IMPLEMENTED("Needs to implement only for gpus");
}

}  // namespace Profiling
}  // namespace Lotus
