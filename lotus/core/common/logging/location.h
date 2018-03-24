#pragma once

#include <sstream>
#include <string>

namespace Lotus {
namespace Logging {
struct Location {
  Location(const char* file_path, const int line, const char* func)
      : file_and_path{file_path}, line_num{line}, function{func} {
  }

  std::string FileNoPath() const {
    // assuming we always have work to do, so not trying to avoid creating a new string if
    // no path was removed.
    return file_and_path.substr(file_and_path.find_last_of("/\\") + 1);
  }

  enum Format {
    kFilename,
    kFilenameAndPath
  };

  std::string ToString(Format format = Format::kFilename) const {
    std::ostringstream out;
    out << (format == Format::kFilename ? FileNoPath() : file_and_path) << ":" << line_num << " " << function;
    return out.str();
  }

  const std::string file_and_path;
  const int line_num;
  const std::string function;
};
}  // namespace Logging
}  // namespace Lotus
