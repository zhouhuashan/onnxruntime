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
  virtual Lotus::Common::Status GetNodeName(std::string* out) = 0;
  //The number of input/output pairs
  virtual size_t GetDataCount() const = 0;
  virtual const onnx::ValueInfoProto& GetOutputInfoFromModel(size_t i) const = 0;
  virtual ~ITestCase() {}
  virtual double GetPerSampleTolerance() const = 0;
  virtual double GetRelativePerSampleTolerance() const = 0;
};

class DataLoder {
 public:
  virtual Lotus::Common::Status Load(const std::experimental::filesystem::v1::path& p, std::unique_ptr<Lotus::MLValue>& value) const = 0;
  virtual ~DataLoder() {}
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
  std::unordered_map<std::string, DataLoder*> loaders;
  std::string test_case_name;
  std::experimental::filesystem::v1::path model_url;
  Lotus::AllocatorPtr allocator_;
  std::vector<onnx::ValueInfoProto> input_value_info_;
  std::vector<onnx::ValueInfoProto> output_value_info_;

  Lotus::Common::Status FromPbFiles(const std::vector<std::experimental::filesystem::v1::path>& files, std::vector<Lotus::MLValue>& output_values);
  std::vector<std::experimental::filesystem::v1::path> test_data_dirs;

  //If we cannot get input name from input_pbs, we'll use names like "data_0","data_1",... It's dirty hack
  // for https://github.com/onnx/onnx/issues/679
  Lotus::Common::Status ConvertInput(const std::vector<onnx::TensorProto>& input_pbs, std::unordered_map<std::string, Lotus::MLValue>& out);
  std::string node_name;
  std::once_flag model_parsed;
  Lotus::Common::Status ParseModel();
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(OnnxTestCase);

 public:
  OnnxTestCase(const Lotus::AllocatorPtr&, const std::string& test_case_name);
  ~OnnxTestCase() {
    for (auto& ivp : loaders) {
      delete ivp.second;
    }
  }
  //TODO: make it configurable
  double GetPerSampleTolerance() const override {
    return 1e-3;
  }
  //TODO: make it configurable
  double GetRelativePerSampleTolerance() const override {
    return 1e-5;
  }
  const onnx::ValueInfoProto& GetOutputInfoFromModel(size_t i) const override {
    return output_value_info_[i];
  }
  size_t GetDataCount() const override {
    return test_data_dirs.size();
  }
  Lotus::Common::Status GetNodeName(std::string* out) override {
    Lotus::Common::Status st = ParseModel();
    if (st.IsOK()) *out = node_name;
    return st;
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
};
