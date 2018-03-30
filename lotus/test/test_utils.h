#pragma once

#include "core/common/logging/logging.h"

namespace Lotus {
namespace Test {

//! Static logging manager with a CLog based sink so logging macros that use the default
//! logger will work
Lotus::Logging::LoggingManager& DefaultLoggingManager();

/**
Perform default initialization of a unit test executable.
This includes setting up google test and the default logger.
*/
void DefaultInitialize(int argc, char** argv, bool create_default_logging_manager = true);

}  // namespace Test
}  // namespace Lotus
