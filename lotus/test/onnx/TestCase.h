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
  virtual Lotus::Common::Status GetPerSampleTolerance(double* value) = 0;
  virtual Lotus::Common::Status GetRelativePerSampleTolerance(double* value) = 0;
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
  std::unordered_map<std::string, DataLoder*> loaders_;
  std::string test_case_name_;
  std::experimental::filesystem::v1::path model_url_;
  Lotus::AllocatorPtr allocator_;
  std::vector<onnx::ValueInfoProto> input_value_info_;
  std::vector<onnx::ValueInfoProto> output_value_info_;

  Lotus::Common::Status FromPbFiles(const std::vector<std::experimental::filesystem::v1::path>& files, std::vector<Lotus::MLValue>& output_values);
  std::vector<std::experimental::filesystem::v1::path> test_data_dirs_;

  //If we cannot get input name from input_pbs, we'll use names like "data_0","data_1",... It's dirty hack
  // for https://github.com/onnx/onnx/issues/679
  Lotus::Common::Status ConvertInput(const std::vector<onnx::TensorProto>& input_pbs, std::unordered_map<std::string, Lotus::MLValue>& out);
  std::string node_name_;
  std::once_flag model_parsed_;
  std::once_flag config_parsed_;
  double per_sample_tolerance_;
  double relative_per_sample_tolerance_;
  Lotus::Common::Status ParseModel();
  Lotus::Common::Status ParseConfig();
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(OnnxTestCase);

 public:
  OnnxTestCase(const std::string& test_case_name) : test_case_name_(test_case_name) {}
  OnnxTestCase(const Lotus::AllocatorPtr&, const std::string& test_case_name);
  ~OnnxTestCase() {
    for (auto& ivp : loaders_) {
      delete ivp.second;
    }
  }
  void SetAllocator(const Lotus::AllocatorPtr&);
  Lotus::Common::Status GetPerSampleTolerance(double* value) override;
  Lotus::Common::Status GetRelativePerSampleTolerance(double* value) override;

  const onnx::ValueInfoProto& GetOutputInfoFromModel(size_t i) const override {
    return output_value_info_[i];
  }
  size_t GetDataCount() const override {
    return test_data_dirs_.size();
  }
  Lotus::Common::Status GetNodeName(std::string* out) override {
    Lotus::Common::Status st = ParseModel();
    if (st.IsOK()) *out = node_name_;
    return st;
  }
  Lotus::Common::Status SetModelPath(const std::experimental::filesystem::v1::path& path) override;

  const std::experimental::filesystem::v1::path& GetModelUrl() const override {
    return model_url_;
  }
  const std::string& GetTestCaseName() const override {
    return test_case_name_;
  }
  Lotus::Common::Status LoadInputData(size_t id, std::unordered_map<std::string, Lotus::MLValue>& feeds) override;
  Lotus::Common::Status LoadOutputData(size_t id, std::vector<Lotus::MLValue>& output_values) override;
};
