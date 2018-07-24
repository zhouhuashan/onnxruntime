#include <Windows.h>
#include <psapi.h>

namespace Lotus {
namespace PerfTest {
namespace Utils {

size_t GetPeakWorkingSetSize() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return pmc.PeakWorkingSetSize;
  }

  return 0;
}

}  // namespace Utils
}  // namespace PerfTest
}  // namespace Lotus