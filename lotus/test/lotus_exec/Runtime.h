//
// Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once

#include "TestDataReader.h"

#include <algorithm>
#include <codecvt>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/framework/inference_session.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#if !defined(_MSC_VER)
#include <sys/stat.h>

#define ERROR_FILE_NOT_FOUND 2L
#define ERROR_BAD_FORMAT 11L

#define O_BINARY 0x0000
#endif

class WinMLRuntime {
 public:
  WinMLRuntime() {
    using namespace Lotus;
    using namespace Lotus::Logging;
    static LoggingManager& s_default_logging_manager = DefaultLoggingManager();
    SessionOptions so;
    so.session_logid = "WinMLRuntime";

    inference_session_ = std::make_unique<Lotus::InferenceSession>(so, &s_default_logging_manager);
  }

  Lotus::Common::Status LoadModel(const std::wstring& model_path) {
    Lotus::Common::Status result = inference_session_->Load(wstr2str(model_path));
    if (result.IsOK())
      result = inference_session_->Initialize();

    return result;
  }

  void FillInBatchSize(std::vector<int64_t>& shape, int input_size, int feature_size) {
    if ((input_size % feature_size != 0) && (feature_size != -1))
      throw std::runtime_error("Input count is not a multiple of dimension.");

    int batch_size = feature_size == -1 ? 1 : input_size / feature_size;
    shape.insert(shape.begin(), batch_size);
  }

  Lotus::MLValue ReadTensorStrings(Lotus::AllocatorPtr alloc, TestDataReader& inputs_reader,
                                   int feature_size, std::vector<int64_t> dims, bool variable_batch_size) {
    using namespace Lotus;

    auto vec = inputs_reader.GetSampleStrings(feature_size, variable_batch_size);

    std::vector<std::string> vec2;
    for (int i = 0; i < vec.size(); i++) {
      std::string str(vec[i].begin(), vec[i].end());
      vec2.push_back(str);
    }

    if (variable_batch_size)
      FillInBatchSize(dims, gsl::narrow_cast<int>(vec.size()), feature_size);

    TensorShape shape(dims);
    auto element_type = DataTypeImpl::GetType<std::string>();

    void* buffer = alloc->Alloc(sizeof(std::string) * shape.Size());
    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                buffer,
                                                                alloc->Info(), alloc);

    std::string* p = p_tensor->MutableData<std::string>();
    for (int i = 0; i < vec.size(); i++) {
      p[i] = std::string(vec[i].begin(), vec[i].end());
    }

    Lotus::MLValue result;
    result.Init(p_tensor.release(),
                DataTypeImpl::GetType<Tensor>(),
                DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

    return result;
  }

  template <typename T>
  Lotus::MLValue ReadTensor(Lotus::AllocatorPtr alloc, TestDataReader& inputs_reader,
                            int feature_size, std::vector<int64_t> dims, bool variable_batch_size) {
    using namespace Lotus;

    auto vec = inputs_reader.GetSample<T>(feature_size, variable_batch_size);

    if (variable_batch_size)
      FillInBatchSize(dims, gsl::narrow_cast<int>(vec.size()), feature_size);

    Lotus::TensorShape shape(dims);
    auto location = alloc->Info();
    auto element_type = Lotus::DataTypeImpl::GetType<T>();
    void* buffer = alloc->Alloc(element_type->Size() * shape.Size());

    if (vec.size() > 0) {
      memcpy(buffer, &vec[0], element_type->Size() * shape.Size());
    }

    std::unique_ptr<Tensor> p_tensor = std::make_unique<Lotus::Tensor>(element_type,
                                                                       shape,
                                                                       buffer,
                                                                       location,
                                                                       alloc);

    Lotus::MLValue result;
    result.Init(p_tensor.release(),
                Lotus::DataTypeImpl::GetType<Lotus::Tensor>(),
                Lotus::DataTypeImpl::GetType<Lotus::Tensor>()->GetDeleteFunc());

    return result;
  }

  template <typename V>
  Lotus::MLValue ReadTensorForMapStringToScalar(Lotus::AllocatorPtr alloc, TestDataReader& inputs_reader) {
    auto vec = inputs_reader.GetSample<V>(-1);

    auto data = std::make_unique<std::map<std::string, V>>();
    for (int i = 0; i < vec.size(); i++) {
      // keys start at "1" so convert index to string key based on that
      data->insert({std::to_string(i + 1), vec[i]});
    }

    Lotus::MLValue result;
    result.Init(data.release(),
                Lotus::DataTypeImpl::GetType<std::map<std::string, V>>(),
                Lotus::DataTypeImpl::GetType<std::map<std::string, V>>()->GetDeleteFunc());

    return result;
  }

  int Run(TestDataReader& inputs_reader) {
    using namespace Lotus;
    int hr = 0;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // Create CPU input tensors
    Lotus::NameMLValMap feed;
    inputs_reader.BufferNextSample();
    if (inputs_reader.Eof())
      return 0;

    bool variable_batch_size = false;
    auto inputs_pairs = inference_session_->GetInputs();
    if (!inputs_pairs.first.IsOK()) {
      auto error = inputs_pairs.first.ErrorMessage();
      return inputs_pairs.first.Code();
    }

    auto& inputs = *(inputs_pairs.second);
    for (size_t index = 0, end = inputs.size(); index < end; ++index) {
      MLValue mlvalue;
      const LotusIR::NodeArg& input = *(inputs[index]);
      const TensorShapeProto* input_shape = input.Shape();
      if (input.Name().empty())
        continue;

      auto type = input.Type();

      std::vector<int64_t> shape;
      int feature_size = -1;

      //Previous graph input was variable length that consumed entire input line_ so fetch new input line_.
      if (variable_batch_size)
        inputs_reader.BufferNextSample();

      //This graph input may or may not be variable length.
      //REVIEW mzs: this can cause issues if we had variable-input followed by fixed input followed by variable-input where
      //fixed-input consumed all of the input line_. *Ideally each graph input should be on its own line_*.
      variable_batch_size = false;

      //If the shape is not available then read everything into the input tensor.
      //feature_size = -1 indicates this condition.
      if (input_shape) {
        feature_size = 0;
        auto dims = input_shape->dim();
        for (auto dim : dims) {
          if (dim.has_dim_param())
            variable_batch_size = true;
          else {
            auto dim_value = dim.dim_value();
            shape.push_back(dim_value);
            feature_size = gsl::narrow_cast<int>(feature_size ? feature_size * dim_value : dim_value);
          }
        }
      }

      //REVIEW mzs: Here an assumption is made that all the input columns are for the map.
      //The supported map types in Lotus seen so far are <string, string> or <string, int64>.
      if (*type == "map(string,tensor(int64))") {
        // check if really map(string, int64), which is all we currently support
        bool is_map_value_scalar = input.TypeAsProto()->map_type().value_type().tensor_type().shape().dim_size() == 0;

        if (is_map_value_scalar) {
          mlvalue = ReadTensorForMapStringToScalar<int64_t>(TestCPUExecutionProvider().GetAllocator(), inputs_reader);
          feed.insert(std::make_pair(input.Name(), mlvalue));
        } else {
          throw std::runtime_error("Unsupported input type: " + std::string(*type));
        }
      } else if (*type == "map(string,tensor(float))" || *type == "map(string,tensor(double))") {
        // check if really map(string, float) or map(string, double), which is all we currently support
        bool is_map_value_scalar = input.TypeAsProto()->map_type().value_type().tensor_type().shape().dim_size() == 0;

        if (is_map_value_scalar) {
          mlvalue = ReadTensorForMapStringToScalar<float>(TestCPUExecutionProvider().GetAllocator(), inputs_reader);
          feed.insert({input.Name(), mlvalue});
        } else {
          throw std::runtime_error("Unsupported input type: " + std::string(*type));
        }
      } else {
        if (*type == "tensor(double)" || *type == "tensor(float)") {
          // If double is used in the following statement, following error occurs.
          // Tensor type mismatch, caller expects elements to be float while tensor contains double Error from operator
          mlvalue = ReadTensor<float>(TestCPUExecutionProvider().GetAllocator(), inputs_reader, feature_size, shape, variable_batch_size);
        } else if (*type == "tensor(int64)")
          mlvalue = ReadTensor<int64_t>(TestCPUExecutionProvider().GetAllocator(), inputs_reader, feature_size, shape, variable_batch_size);
        else if (*type == "tensor(string)")
          mlvalue = ReadTensorStrings(TestCPUExecutionProvider().GetAllocator(), inputs_reader, feature_size, shape, variable_batch_size);
        else
          throw std::runtime_error("Unsupported input type: " + std::string(*type));

        feed.insert(std::make_pair(input.Name(), mlvalue));
      }
    }

    // Create output feed
    std::vector<std::string> output_names;
    for (auto const& outp : *(inference_session_->GetOutputs().second)) {
      output_names.push_back(outp->Name());
    }

    std::cout.precision(12);
    std::string separator = "";
    // Invoke the net
    std::vector<Lotus::MLValue> outputMLValue;
    Lotus::Common::Status result = inference_session_->Run(feed, &outputMLValue);
    if (result.IsOK()) {
      auto outputMeta = inference_session_->GetOutputs().second;
      // Peel the data off the CPU
      for (unsigned int i = 0; i < output_names.size(); i++) {
        Lotus::MLValue& output = outputMLValue[i];
        const Lotus::Tensor* ctensor = nullptr;

        if (output.IsTensor()) {
          ctensor = &output.Get<Tensor>();
          //REVIEW mzs: Map output types are not tested because I couldn't find any tests for that.
          if (ctensor->DataType() == Lotus::DataTypeImpl::GetType<std::map<int64_t, float>>()) {
            const std::map<int64_t, float>* ci = &output.Get<std::map<int64_t, float>>();
            for (const auto& p : *ci) {
              std::cout << separator << p.second;
              separator = ",";
            }
          } else if (ctensor->DataType() == Lotus::DataTypeImpl::GetType<std::map<std::string, float>>()) {
            const std::map<std::string, float>* ci = &output.Get<std::map<std::string, float>>();
            for (const auto& p : *ci) {
              std::cout << separator << p.second;
              separator = ",";
            }
          } else if (ctensor->DataType() == Lotus::DataTypeImpl::GetType<float>()) {
            const float* cdata = ctensor->Data<float>();
            for (int ci = 0; ci < ctensor->Shape().Size(); ci++) {
              std::cout << separator << cdata[ci];
              separator = ",";
            }
          } else if (ctensor->DataType() == Lotus::DataTypeImpl::GetType<int64_t>()) {
            const int64_t* cdata = ctensor->Data<int64_t>();
            for (int ci = 0; ci < ctensor->Shape().Size(); ci++) {
              std::cout << separator << cdata[ci];
              separator = ",";
            }
          } else if (ctensor->DataType() == Lotus::DataTypeImpl::GetType<std::string>()) {
            const std::string* cdata = ctensor->Data<std::string>();
            for (int ci = 0; ci < ctensor->Shape().Size(); ci++) {
              std::cout << separator << cdata[ci];
              separator = ",";
            }
          } else {
            throw std::runtime_error("Unsupported output type in Lotus model: " + std::string((*outputMeta)[i]->Name()));
          }
        } else if (output.Type() == Lotus::DataTypeImpl::GetType<Lotus::VectorMapStringToFloat>()) {
          auto& cdata = output.Get<Lotus::VectorMapStringToFloat>();
          for (int ci = 0; ci < cdata.size(); ci++) {
            for (const auto& p : cdata[ci]) {
              std::cout << separator << p.second;
              separator = ",";
            }
          }
        } else if (output.Type() == Lotus::DataTypeImpl::GetType<Lotus::VectorMapInt64ToFloat>()) {
          auto& cdata = output.Get<Lotus::VectorMapInt64ToFloat>();
          for (int ci = 0; ci < cdata.size(); ci++) {
            for (const auto& p : cdata[ci]) {
              std::cout << separator << p.second;
              separator = ",";
            }
          }
        }
      }

      std::cout << std::endl;
    } else {
      std::cerr << result.ErrorMessage() << std::endl;
      hr = result.Code();
    }

    return hr;
  }

 private:
  std::unique_ptr<Lotus::InferenceSession> inference_session_;

  static Lotus::Logging::LoggingManager& DefaultLoggingManager() {
    using namespace Lotus;
    std::string default_logger_id{"Default"};

    static Logging::LoggingManager default_logging_manager{
        std::unique_ptr<Logging::ISink>{new Lotus::Logging::CLogSink{}},
        Logging::Severity::kWARNING, false,
        Logging::LoggingManager::InstanceType::Default,
        &default_logger_id};

    return default_logging_manager;
  }

  static Lotus::IExecutionProvider& TestCPUExecutionProvider() {
    static Lotus::CPUExecutionProviderInfo info;
    static Lotus::CPUExecutionProvider cpu_provider(info);
    return cpu_provider;
  }
};
