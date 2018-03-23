#pragma once

#include <ostream>
#include <sstream>
#include <string>

#include "date/date.h"

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"

namespace Lotus {
namespace Logging {
/// <summary>
/// A std::ostream based ISink
/// </summary>
/// <seealso cref="ISink" />
class OStreamSink : public ISink {
 protected:
  OStreamSink(std::ostream &stream, bool flush)
      : stream_{&stream}, flush_{flush} {
  }

 public:
  void SendImpl(const Timestamp &timestamp, const std::string &logger_id, const Capture &message) override {
    // operator for formatting of timestamp in ISO8601 format including microseconds
    using date::operator<<;

    // Two options as there may be multiple calls attempting to write to the same sink at once:
    // 1) Use mutex to synchronize access to the stream.
    // 2) Create the message in an ostringstream and output in one call.
    //
    // Going with #2 as it should scale better at the cost of creating the message in memory first
    // before sending to the stream.

    std::ostringstream msg;

    msg << timestamp << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
        << message.Location().ToString() << "] " << message.Message();

    (*stream_) << msg.str() << "\n";

    if (flush_) {
      stream_->flush();
    }
  }

 private:
  std::ostream *stream_;
  const bool flush_;
};
}  // namespace Logging
}  // namespace Lotus
