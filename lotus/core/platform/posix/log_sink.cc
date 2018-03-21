// @@COPYRIGHT@@

#include "core/common/log_sink.h"

namespace Lotus {
LogSinkPtr GetDefaultLogSink() {
#ifdef LOTUS_ENABLE_STDERR_LOGGING
  return std::make_unique<StdErrLogSink>();
#else
  return nullptr;
#endif
}
}  // namespace Lotus
