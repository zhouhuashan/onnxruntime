#pragma once

#include <iostream>
#include "core/common/logging/sinks/ostream_sink.h"

namespace Lotus {
namespace Logging {
/// <summary>
/// A std::clog based ISink
/// </summary>
/// <seealso cref="ISink" />
class CLogSink : public OStreamSink {
 public:
  CLogSink() : OStreamSink(std::clog, /*flush*/ true) {
  }
};
}  // namespace Logging
}  // namespace Lotus
