#include "core/common/logging/isink.h"

class VsTestSink : public Lotus::Logging::ISink {
 public:
  void SendImpl(const Lotus::Logging::Timestamp &timestamp, const std::string &logger_id_, const Lotus::Logging::Capture &message) override;
};