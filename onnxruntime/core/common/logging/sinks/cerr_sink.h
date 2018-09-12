#pragma once

#include <iostream>
#include "core/common/logging/sinks/ostream_sink.h"

namespace onnxruntime {
namespace Logging {
/// <summary>
/// A std::cerr based ISink
/// </summary>
/// <seealso cref="ISink" />
class CErrSink : public OStreamSink {
 public:
  CErrSink() : OStreamSink(std::cerr, /*flush*/ false) {  // std::cerr isn't buffered so no flush required
  }
};
}  // namespace Logging
}  // namespace onnxruntime
