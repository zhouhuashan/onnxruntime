#pragma once

#include <iostream>
#if defined(_MSC_VER)
#include <filesystem>
#else
#include <experimental/filesystem>
// HRESULT is a 4-byte long on MSVC.  We'll just make it a signed int here.
typedef int HRESULT;
// Success codes
#define S_OK ((HRESULT)0L)
#define S_FALSE ((HRESULT)1L)
#endif
#include "Runtime.h"
#include "CmdParser.h"

enum ExecutionStatus {
  OK = 0,
  MODEL_LOADING_FAILURE = 1,
  DATA_LOADING_FAILURE = 2,
  PREDICTION_FAILURE = 3,
  NOT_IMPLEMENTED = 4
};

class Model {
 public:
  Model(const std::string &modelfile) {
    try {
      runtime_ = LoadModel(modelfile);
    } catch (...) {
      runtime_ = nullptr;
    }

    if (!runtime_) {
      CleanUp();
      execStatus = ExecutionStatus::MODEL_LOADING_FAILURE;
      return;
    }
  }

  void Execute(const std::string &datafile) {
    struct stat s;
    if (stat(datafile.c_str(), &s) == 0) {
      if (s.st_mode & S_IFDIR) {
        execStatus = ExecutionStatus::NOT_IMPLEMENTED;
        CleanUp();
        return;
      }
    }

    auto input_reader = std::unique_ptr<TestDataReader>(LoadTestFile(datafile));
    if (!input_reader) {
      execStatus = ExecutionStatus::DATA_LOADING_FAILURE;
      CleanUp();
      return;
    }

    int sample = 0;
    while (!input_reader->Eof()) {
      std::map<std::string, std::vector<float>> outputs;

      // Perform the test
      int hr = runtime_->Run(*input_reader);
      if (hr != 0) {
        std::cerr << "Failed to execute example" << std::endl;
        execStatus = ExecutionStatus::PREDICTION_FAILURE;
        return;
      }
      sample++;
    }
  }

  ExecutionStatus GetStatus() {
    return execStatus;
  }

  std::string GetStatusString() {
    switch (execStatus) {
      case ExecutionStatus::OK:
        return "OK";
      case ExecutionStatus::MODEL_LOADING_FAILURE:
        return "MODEL_LOADING_FAILURE";
      case ExecutionStatus::DATA_LOADING_FAILURE:
        return "DATA_LOADING_FAILURE";
      case ExecutionStatus::PREDICTION_FAILURE:
        return "PREDICTION_FAILURE";
      default:
        return "UNKNOWN";
    }
  }

  ~Model() {
    CleanUp();
  }

 private:
  void CleanUp() {
    if (runtime_) {
      delete (runtime_);
      runtime_ = nullptr;
    }
  }

  WinMLRuntime *LoadModel(const std::string &strfilepath) {
    std::wstring filepath(strfilepath.begin(), strfilepath.end());
    auto runtime = new WinMLRuntime();
    std::string error_message;
    if (runtime->LoadModel(filepath, error_message) == 0) {
      std::cerr << "'" << strfilepath.c_str() << "' loaded successfully." << std::endl;
    } else {
      std::cerr << "Loading failed for '" << strfilepath.c_str() << "'" << std::endl;
      std::cerr << "-----------------------------" << std::endl;
      std::cerr << error_message << std::endl;
      return nullptr;
    }
    return runtime;
  }

  TestDataReader *LoadTestFile(const std::string &filepath) {
    std::wstring testfilepath(filepath.begin(), filepath.end());
    auto reader = TestDataReader::OpenReader(testfilepath);

    if (!reader) {
      std::cerr << "Unable to load test data file." << std::endl;
      return nullptr;
    }
    return reader;
  }

  WinMLRuntime *runtime_ = nullptr;
  ExecutionStatus execStatus = ExecutionStatus::OK;
};
