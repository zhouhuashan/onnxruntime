#pragma once

#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif
#include <fstream>

struct TestCaseInfo {
  std::string model_url;
  std::string test_case_name;
  bool is_single_noded = false;
  bool is_implemented = false;
  std::vector<std::vector<std::experimental::filesystem::v1::path> > input_pb_files;
  std::vector<std::vector<std::experimental::filesystem::v1::path> > output_pb_files;
};