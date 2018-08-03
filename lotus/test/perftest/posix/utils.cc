#include "test/perftest/utils.h"

#include <cstddef>

#include <sys/time.h>
#include <sys/resource.h>

namespace Lotus {
namespace PerfTest {
namespace Utils {

std::size_t GetPeakWorkingSetSize() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return static_cast<size_t>(rusage.ru_maxrss * 1024L);
}

class CPUUsage : public ICPUUsage {
 public:
  CPUUsage() {
    Reset();
  }

  short GetUsage() const override {
     // To Be Impelemented.
    return 0;
  }

  void Reset() override {
     // To Be Implemented.
  }
};

std::unique_ptr<ICPUUsage> CreateICPUUsage() {
  return std::make_unique<CPUUsage>();
}

}  // namespace Utils
}  // namespace PerfTest
}  // namespace Lotus