#include "core/framework/inference_session.h"
#include "gtest/gtest.h"

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#else
#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

namespace Lotus {
namespace Test {

class LotusModelTest : public testing::Test {
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }

 public:
  // run test using LotusRT::Run(const std::vector<std::string>& output_names ...) api
  template <typename Tin, typename Tout>
  void RunLotusModelRun(const std::string name, const std::map<std::string, std::pair<std::vector<int64_t>, std::vector<Tin> > > inputs,
                        const std::string output_name, const std::vector<int64_t> out_dims, const std::vector<Tout> expected_output,
                        const SessionOptions& options) {
    InferenceSession session_object{options};
    EXPECT_TRUE(session_object.Load(name).IsOK());
    EXPECT_TRUE(session_object.Initialize().IsOK());

    NameMLValMap feeds;
    std::vector<std::string> output_names;
    for (auto it = inputs.begin(); it != inputs.end(); it++) {
      MLValue value;
      EXPECT_TRUE(CreateMLValue<Tin>(it->second.first, it->second.second, &value).IsOK());
      feeds[it->first] = value;
    }
    output_names.push_back(output_name);
    std::vector<MLValue> fetches;
    RunOptions run_options;
    //run_options.timeout_in_ms = 0;
    run_options.run_tag = "one session/one thread";
    Status st = session_object.Run(run_options, feeds, output_names, &fetches);
    EXPECT_TRUE(st.IsOK());
    ASSERT_EQ(1, fetches.size());
    auto& rtensor = fetches.front().Get<Tensor>();
    TensorShape expected_shape(out_dims);
    EXPECT_EQ(expected_shape, rtensor.Shape());

    const std::vector<Tout> found(rtensor.Data<Tout>(), rtensor.Data<Tout>() + expected_shape.Size());
    ASSERT_EQ(expected_output, found);
  }

  template <typename T>
  Status CreateMLValue(const std::vector<int64_t>& dims, const std::vector<T>& value, MLValue* output,
                       IAllocator* alloc = nullptr) {
    if (!alloc) {
      alloc = &AllocatorManager::Instance().GetArena(CPU);
    }
    TensorShape shape(dims);
    auto location = alloc->Info();
    auto element_type = DataTypeImpl::GetType<T>();
    void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
    if (value.size() > 0) {
      memcpy(buffer, &value[0], element_type->Size() * shape.Size());
    }

    Tensor* tensor = new Tensor(
        element_type,
        shape,
        std::move(BufferUniquePtr(buffer, BufferDeleter(alloc))),
        location);
    output->Init(tensor,
                 DataTypeImpl::GetType<Tensor>(),
                 DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    return Status::OK();
  }

 protected:
  const std::string current_exe_dir_ = ".";

  const std::string squeezenet_ = current_exe_dir_ + "/testdata/squeezenet/model.onnx";
  const std::string squeezenet_input_0 = current_exe_dir_ + "/testdata/squeezenet/test_data_set_0/test_data_0_input.pb";
  const std::string squeezenet_output_0 = current_exe_dir_ + "/testdata/squeezenet/test_data_set_0/test_data_0_output.pb";

  const std::vector<int64_t> dims_mul_x_ = {3, 2};
  const std::vector<int64_t> dims_mul_y_ = {3, 2};
  const std::vector<float> values_mul_ = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> expect_mul_ = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  int FdOpen(const std::string& name) {
    int fd = -1;
#ifdef _WIN32
    _sopen_s(&fd, name.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
    fd = open(name.c_str(), O_RDONLY);
#endif
    return fd;
  };

  void FdClose(int fd) {
    if (fd >= 0) {
#ifdef _WIN32
      _close(fd);
#else
      close(fd);
#endif
    }
  };

  bool LoadTensorFromPb(onnx::TensorProto& tensor, std::string file) {
    int fd = FdOpen(file);
    if (fd < 0) {
      return false;
    }
    if (!tensor.ParseFromFileDescriptor(fd)) {
      FdClose(fd);
      return false;
    }
    FdClose(fd);
    return true;
  }
};

TEST_F(LotusModelTest, squeeze_net) {
  ExecutionProviderInfo epi;
  ProviderOption po{"CPUExecutionProvider", epi};
  SessionOptions so(vector<ProviderOption>{po});

  onnx::TensorProto input;
  EXPECT_TRUE(LoadTensorFromPb(input, squeezenet_input_0));

  onnx::TensorProto output;
  EXPECT_TRUE(LoadTensorFromPb(output, squeezenet_output_0));

  std::vector<float> in(input.float_data().begin(), input.float_data().end());
  std::vector<float> out(output.float_data().begin(), output.float_data().end());
  std::vector<int64_t> shape, output_shape;
  shape.assign(input.dims().begin(), input.dims().end());
  output_shape.assign(output.dims().begin(), output.dims().end());

  RunLotusModelRun<float, float>(squeezenet_, {{"data_0", {shape, in}}},
                                 std::string("softmaxout_1"), output_shape, out, so);
}

}  // namespace Test
}  // namespace Lotus
