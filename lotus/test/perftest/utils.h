#pragma once

#include <memory>

namespace Lotus {
namespace PerfTest {
namespace Utils{

size_t GetPeakWorkingSetSize();

class ICPUUsage{
 public:
  virtual ~ICPUUsage() = default;

  virtual short GetUsage() const = 0;
  
  virtual void Reset() = 0;
};

std::unique_ptr<ICPUUsage> CreateICPUUsage();

}  // namespace Utils
}  // namespace PerfTest
}  // namespace Lotus
