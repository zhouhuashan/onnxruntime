#pragma once

#include <string>

#include "core/common/logging/logging.h"

namespace Lotus {
namespace Logging {
class ISink {
 public:
  /**
  Sends the message to the sink.
  @param timestamp The timestamp.
  @param logger_id The logger identifier.
  @param message The captured message.
  */
  void Send(const Timestamp &timestamp, const std::string &logger_id, const Capture &message) {
    SendImpl(timestamp, logger_id, message);
  }

  virtual ~ISink() = default;

 private:
  virtual void SendImpl(const Timestamp &timestamp, const std::string &logger_id, const Capture &message) = 0;
};
}  // namespace Logging
}  // namespace Lotus
