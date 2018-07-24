#include <cstddef>

#include <sys/time.h>
#include <sys/resource.h>

namespace Lotus {
namespace PerfTest {
namespace Utils {

std::size_t GetPeakWorkingSetSize() {
   //TO DO: add support for other linux version
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return static_cast<size_t>(rusage.ru_maxrss * 1024L);
}

}  // namespace Utils
}  // namespace PerfTest
}  // namespace Lotus