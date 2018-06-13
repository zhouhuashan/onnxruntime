#pragma once
#include <vector>
#include <core/framework/ml_value.h>

#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif

//One test case is for one model file
//One test case can contain multiple test data
class ITestCase {
 public:
  //must be called before calling the other functions
  virtual Lotus::Common::Status SetModelPath(const std::experimental::filesystem::v1::path& path) = 0;
  virtual Lotus::Common::Status LoadInputData(size_t id, std::unordered_map<std::string, Lotus::MLValue>& feeds) = 0;
  virtual Lotus::Common::Status LoadOutputData(size_t id, std::vector<Lotus::MLValue>& output_values) = 0;
  virtual const std::experimental::filesystem::v1::path& GetModelUrl() const = 0;
  virtual const std::string& GetTestCaseName() const = 0;
  virtual const std::string& GetNodeName() const = 0;
  //The number of input/output pairs
  virtual size_t GetDataCount() const = 0;
  virtual ~ITestCase(){};
};

/**
* test_case_dir must have contents of:
* model.onnx
* ???/input_??.pb
* ???/output_??.pb
* ???/input_??.pb
* ???/output_??.pb
*/
class OnnxTestCase : public ITestCase {
 private:
  std::string test_case_name;
  std::experimental::filesystem::v1::path model_url;
  Lotus::AllocatorPtr allocator_;
  std::vector<onnx::ValueInfoProto> input_value_info_;
  Lotus::Common::Status FromTensorProto(const onnx::TensorProto& input, std::unique_ptr<Lotus::MLValue>& value);
  Lotus::Common::Status FromTensorProto(const std::vector<onnx::TensorProto>& input, std::vector<Lotus::MLValue>& output_values);
  Lotus::Common::Status FromPbFiles(const std::vector<std::experimental::filesystem::v1::path>& files, std::vector<Lotus::MLValue>& output_values);
  std::vector<std::experimental::filesystem::v1::path> test_data_dirs;

  //If we cannot get input name from input_pbs, we'll use names like "data_0","data_1",... It's dirty hack
  // for https://github.com/onnx/onnx/issues/679
  Lotus::Common::Status ConvertInput(const std::vector<onnx::TensorProto>& input_pbs, std::unordered_map<std::string, Lotus::MLValue>& out);
  std::string node_name;

 public:
  size_t GetDataCount() const override {
    return test_data_dirs.size();
  }
  const std::string& GetNodeName() const override {
    return node_name;
  }
  Lotus::Common::Status SetModelPath(const std::experimental::filesystem::v1::path& path) override;

  const std::experimental::filesystem::v1::path& GetModelUrl() const override {
    return model_url;
  }
  const std::string& GetTestCaseName() const override {
    return test_case_name;
  }
  Lotus::Common::Status LoadInputData(size_t id, std::unordered_map<std::string, Lotus::MLValue>& feeds) override;
  Lotus::Common::Status LoadOutputData(size_t id, std::vector<Lotus::MLValue>& output_values) override;
  OnnxTestCase(Lotus::AllocatorPtr allocator, const std::string& test_case_name);
};
