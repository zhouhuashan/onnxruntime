#pragma once

#include <cstdarg>

#include "core/common/common.h"
#include "core/common/logging/location.h"
#include "core/common/logging/logging.h"

namespace Lotus {
namespace Logging {
class Logger;

/// <summary>
/// Class to capture the details of a log message.
/// </summary>
class Capture {
 public:
  /// <summary>
  /// Initializes a new instance of the <see cref="Capture"/> class.
  /// </summary>
  /// <param name="logger">The logger.</param>
  /// <param name="severity">The severity.</param>
  /// <param name="category">The category.</param>
  /// <param name="dataType">Type of the data.</param>
  /// <param name="location">The file location the log message is coming from.</param>
  Capture(const Logger *logger, Logging::Severity severity, const char *category,
          Logging::DataType dataType, const Location &location)
      : logger_{logger}, severity_{severity}, category_{category}, data_type_{dataType}, location_{location} {
  }

  /// <summary>
  /// The stream that can capture the message via operator<<.
  /// </summary>
  /// <returns>Output stream.</returns>
  std::ostream &Stream() noexcept {
    return stream_;
  }

#ifdef _MSC_VER
// add SAL annotation for printf format string. requires Code Analysis to run to validate usage.
#define msvc_printf_check _Printf_format_string_
#define __attribute__(x)  // Disable for MSVC. Supported by GCC and CLang.
#else
#define msvc_printf_check
#endif

  /// <summary>
  /// Captures a printf style log message.
  /// </summary>
  /// <param name="format">The printf format.</param>
  /// <param name="">Arguments to the printf format if needed.</param>
  /// <remarks>
  /// A maximum of 2K of output will be captured currently.
  /// Non-static method, so 'this' is implicit first arg, and we use format(printf(2,3)
  /// </remarks>
  void CapturePrintf(msvc_printf_check const char *format, ...) __attribute__((format(printf, 2, 3)));

  /// <summary>
  /// Captures a printf style log message.
  /// </summary>
  /// <param name="format">The printf format.</param>
  /// <param name="">Arguments to the printf format if needed.</param>
  /// <remarks>
  /// A maximum of 2K of output will be captured currently.
  /// </remarks>
  void CapturePrintf(msvc_printf_check const char *format, va_list args);

  Logging::Severity Severity() const noexcept {
    return severity_;
  }

  const char SeverityPrefix() const noexcept {
    return Logging::SEVERITY_PREFIX[static_cast<int>(severity_)];
  }

  const char *Category() const noexcept {
    return category_;
  }

  const Logging::DataType DataType() const noexcept {
    return data_type_;
  }

  const Logging::Location &Location() const noexcept {
    return location_;
  }

  std::string Message() const noexcept {
    return stream_.str();
  }

  ~Capture();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(Capture);

  const Logger *logger_;
  const Logging::Severity severity_;
  const char *category_;
  const Logging::DataType data_type_;
  const Logging::Location location_;

  std::ostringstream stream_;
};
}  // namespace Logging
}  // namespace Lotus
