#pragma once

// __PRETTY_FUNCTION__ isn't a macro on gcc, so use a check for _MSC_VER
// so we only define it as one for MSVC
#if _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif

// Capture where the message is coming from
#define WHERE \
  Lotus::Logging::Location(__FILE__, __LINE__, __PRETTY_FUNCTION__)

#define CREATE_MESSAGE(logger_ptr, severity, category, datatype) \
  Lotus::Logging::Capture(logger_ptr, Lotus::Logging::Severity::k##severity, category, datatype, WHERE)

/*
Both printf and stream style logging are supported.
Not that printf currently has a 2K limit to the message size.

LOGS_* macros are for stream style
LOGF_* macros are for printf style

The Message class captures the log input, and pushes it through the logger in its destructor.

Use the *FATAL* macros if you want a Severity::kFatal message to also throw.
*/

// iostream style logging. Capture log info in Message, and push to the logger_ptr in ~Message.
#define LOGS(logger_ptr, severity, category)                                                                  \
  if ((logger_ptr)->OutputIsEnabled(Lotus::Logging::Severity::k##severity, Lotus::Logging::DataType::SYSTEM)) \
  CREATE_MESSAGE(logger_ptr, severity, category, Lotus::Logging::DataType::SYSTEM).Stream()

#define LOGS_USER(logger_ptr, severity, category)                                                           \
  if ((logger_ptr)->OutputIsEnabled(Lotus::Logging::Severity::k##severity, Lotus::Logging::DataType::USER)) \
  CREATE_MESSAGE(logger_ptr, severity, category, Lotus::Logging::DataType::USER).Stream()

// printf style logging. Capture log info in Message, and push to the logger_ptr in ~Message.
#define LOGF(logger_ptr, severity, category, format_str, ...)                                                 \
  if ((logger_ptr)->OutputIsEnabled(Lotus::Logging::Severity::k##severity, Lotus::Logging::DataType::SYSTEM)) \
  CREATE_MESSAGE(logger_ptr, severity, category, Lotus::Logging::DataType::SYSTEM).CapturePrintf(format_str, ##__VA_ARGS__)

#define LOGF_USER(logger_ptr, severity, category, format_str, ...)                                          \
  if ((logger_ptr)->OutputIsEnabled(Lotus::Logging::Severity::k##severity, Lotus::Logging::DataType::USER)) \
  CREATE_MESSAGE(logger_ptr, severity, category, Lotus::Logging::DataType::USER).CapturePrintf(format_str, ##__VA_ARGS__)

/*

Macros that use the default logger. 
A LoggingManager instance must be currently valid for the default logger to be available.

*/
#define LOGS_DEFAULT(severity, category) \
  LOGS(Lotus::Logging::LoggingManager::DefaultLogger(), severity, category)

#define LOGS_USER_DEFAULT(severity, category) \
  LOGS_USER(Lotus::Logging::LoggingManager::DefaultLogger(), severity, category)

#define LOGF_DEFAULT(severity, category, format_str, ...) \
  LOGF(Lotus::Logging::LoggingManager::DefaultLogger(), severity, category, format_str, __VA_ARGS__)

#define LOGF_USER_DEFAULT(severity, category, format_str, ...) \
  LOGF_USER(Lotus::Logging::LoggingManager::DefaultLogger(), severity, category, format_str, __VA_ARGS__)

/*

Conditional logging

*/
#define LOGS_IF(boolean_expression, logger_ptr, severity, category) \
  if ((boolean_expression) == true)                                 \
  LOGS(logger_ptr, severity, category)

#define LOGS_DEFAULT_IF(boolean_expression, severity, category) \
  if ((boolean_expression) == true)                             \
  LOGS_DEFAULT(severity, category)

#define LOGS_USER_IF(boolean_expression, logger_ptr, severity, category) \
  if ((boolean_expression) == true)                                      \
  LOGS_USER(logger_ptr, severity, category)

#define LOGS_USER_DEFAULT_IF(boolean_expression, severity, category) \
  if ((boolean_expression) == true)                                  \
  LOGS_USER_DEFAULT(severity, category)

#define LOGF_IF(boolean_expression, logger_ptr, severity, category, format_str, ...) \
  if ((boolean_expression) == true)                                                  \
  LOGF(logger_ptr, severity, category, format_str, __VA_ARGS__)

#define LOGF_DEFAULT_IF(boolean_expression, severity, category, format_str, ...) \
  if ((boolean_expression) == true)                                              \
  LOGF_DEFAULT(severity, category, format_str, __VA_ARGS__)

#define LOGF_USER_IF(boolean_expression, logger_ptr, severity, category, format_str, ...) \
  if ((boolean_expression) == true)                                                       \
  LOGF_USER(logger_ptr, severity, category, format_str, __VA_ARGS__)

#define LOGF_USER_DEFAULT_IF(boolean_expression, severity, category, format_str, ...) \
  if ((boolean_expression) == true)                                                   \
  LOGF_USER_DEFAULT(severity, category, format_str, __VA_ARGS__)

// TODO: Consider alternative.
// The below *FATAL* macros are more testable in that they throw via LogFatalAndThrow.
// Alternatively, LoggingManager can throw after sending a Severity::kFatal message
// as that happens in the context of Capture destructor, leading to std::terminate being
// called. That behavior is more consistent (using kFatal with any log statement will exit)
// however possibly harder to debug, especially if the log message with the fatal error
// hasn't been flushed to the sink when the terminate occurs. Throwing here and catching in a
// top level handler has cleaner unwind semantics, even if that simply re-throws.

// Log at Severity::Fatal and throw.
#define LOGF_FATAL(category, format_str, ...) \
  Lotus::Logging::LoggingManager::LogFatalAndThrow(category, WHERE, format_str, ##__VA_ARGS__)

// If condition is true, log the condition at Severity::Fatal and throw
#define FATAL_IF(boolean_expression) \
  if (boolean_expression)            \
  LOGF_FATAL(Lotus::Logging::Category::Lotus, #boolean_expression)

/*

Debug verbose logging of caller provided level that uses the default logger.
Use the _USER variants for VLOG statements involving user data that may need to be filtered.

*/
#define VLOGS(level)                          \
  if (level < Lotus::Logging::max_vlog_level) \
  LOGS_DEFAULT(VERBOSE, "VLOG" #level)

#define VLOGS_USER(level)                     \
  if (level < Lotus::Logging::max_vlog_level) \
  LOGS_USER_DEFAULT(VERBOSE, "VLOG" #level)

#define VLOGF(level, format_str, ...)         \
  if (level < Lotus::Logging::max_vlog_level) \
  LOGF_DEFAULT(VERBOSE, "VLOG" #level, format_str, __VA_ARGS__)

#define VLOGF_USER(level, format_str, ...)    \
  if (level < Lotus::Logging::max_vlog_level) \
  LOGF_USER_DEFAULT(VERBOSE, "VLOG" #level, format_str, __VA_ARGS__)

/* 
Check macros
*/

#define CHECK_NOTNULL(ptr) \
  FATAL_IF((ptr) == nullptr)

// kFatal if the check does not pass
#define CHECK_OP(val1, val2, op) \
  FATAL_IF(!(val1 op val2))

#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
