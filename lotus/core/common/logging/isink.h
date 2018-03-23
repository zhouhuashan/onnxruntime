#pragma once

#include <string>

#include "core/common/logging/logging.h"

namespace Lotus {
namespace Logging {
class ISink {
 public:
  /// <summary>
  /// Sends the message to the sink.
  /// </summary>
  /// <param name="timestamp">The timestamp.</param>
  /// <param name="logger_id">The logger identifier.</param>
  /// <param name="message">The captured message.</param>
  void Send(const Timestamp &timestamp, const std::string &logger_id, const Capture &message) {
    SendImpl(timestamp, logger_id, message);
  }

  virtual ~ISink() = default;

 private:
  virtual void SendImpl(const Timestamp &timestamp, const std::string &logger_id, const Capture &message) = 0;
};
}  // namespace Logging
}  // namespace Lotus
