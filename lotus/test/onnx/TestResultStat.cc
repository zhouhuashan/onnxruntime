#include <stdio.h>
#include <sstream>
#include <algorithm>
#include "TestResultStat.h"

namespace {
template <typename T1>
std::string containerToStr(const T1& input) {
  std::ostringstream oss;
  bool is_first = true;
  std::vector<typename T1::value_type> vec(input.begin(), input.end());
  std::sort(vec.begin(), vec.end());
  for (const auto& s : vec) {
    if (!is_first) oss << ", ";
    oss << s;
    is_first = false;
  }
  return oss.str();
}
}  // namespace

void TestResultStat::print(const std::vector<std::string>& all_implemented_ops, bool no_coverage_info, FILE* output) {
  std::unordered_set<std::string> succeeded_kernels(this->covered_ops);
  std::vector<std::string> not_tested;
  for (const std::string& s : all_implemented_ops) {
    if (this->covered_ops.find(s) == this->covered_ops.end())
      not_tested.push_back(s);
  }
  for (const std::string& name : this->not_implemented_kernels) {
    succeeded_kernels.erase(name);
  }
  for (const std::string& name : this->failed_kernels) {
    succeeded_kernels.erase(name);
  }
  std::string not_implemented_kernels_str = containerToStr(this->not_implemented_kernels);
  std::string failed_kernels_str = containerToStr(this->failed_kernels);
  std::string not_tested_str = containerToStr(not_tested);
  int failed = (int)this->total_test_case_count - this->succeeded - this->skipped - this->not_implemented;
  int other_reason_failed = failed - this->load_model_failed - this->result_differs - this->throwed_exception;
  std::ostringstream oss;
  oss << "result: \n"
         "\tTotal test cases:"
      << this->total_test_case_count
      << "\n\t\tSucceeded:" << this->succeeded
      << "\n\t\tSkipped:" << (this->skipped + this->not_implemented)
      << "\n\t\t\tKernel not implemented:" << this->not_implemented
      << "\n\t\tFailed:" << failed
      << "\n\t\t\tLoad model Failed:" << this->load_model_failed
      << "\n\t\t\tThrew exception while runnning:" << this->throwed_exception
      << "\n\t\t\tResult differs:" << this->result_differs << "\n";
  if (other_reason_failed != 0) oss << "\t\t\tOther reason:" << other_reason_failed << "\n";
  oss << "\tStats by Operator type:\n";
  if (!no_coverage_info) {
    oss << "\t\tImplemented:" << all_implemented_ops.size() << "\n\t\tNot covered by any test("
        << not_tested.size() << "): " << not_tested_str << "\n\t\tCovered:"
        << this->covered_ops.size() << "\n\t\tSucceeded:"
        << succeeded_kernels.size() << "\n";
  }
  oss << "\t\tNot implemented(" << this->not_implemented_kernels.size() << "): " << not_implemented_kernels_str << "\n\t\tFailed:"
      << failed_kernels_str << "\n";
  std::string res = oss.str();
  fwrite(res.c_str(), 1, res.size(), output);
}