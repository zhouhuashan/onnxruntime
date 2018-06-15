//
// Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once

#include "core/framework/inference_session.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#include <iostream>
#include <vector>
#include <memory>
#include <codecvt>
#include <string>
#include <map>
#include "TestDataReader.h"

#if !defined(_MSC_VER)
#include <sys/stat.h>

#define ERROR_FILE_NOT_FOUND 2L
#define ERROR_BAD_FORMAT 11L

#define O_BINARY 0x0000

typedef signed long INT64;
#endif

class WinMLRuntime {
 public:
  static Lotus::Logging::LoggingManager& GetDefaultLogM() {
    using namespace Lotus;
    std::string default_logger_id{"Default"};

    static Logging::LoggingManager default_logging_manager{std::unique_ptr<Logging::ISink>{new Lotus::Logging::CLogSink{}},
                                                           Logging::Severity::kWARNING, false,
                                                           Logging::LoggingManager::InstanceType::Default,
                                                           &default_logger_id};

    return default_logging_manager;
  }

  WinMLRuntime() {
    using namespace Lotus;
    using namespace Lotus::Logging;
    static LoggingManager* s_default_logging_manager = &GetDefaultLogM();
    SessionOptions so;
    so.session_logid = "InferenceSessionTests.NoTimeout";

    pRuntime = new Lotus::InferenceSession(so, s_default_logging_manager);
  }

  ~WinMLRuntime() {
    if (pRuntime) {
      delete (pRuntime);
      pRuntime = nullptr;
    }
  }

  int LoadModel(std::wstring modelPath, std::string& error) {
    Lotus::Common::Status result = pRuntime->Load(wstr2str(modelPath));
    if (!result.IsOK()) {
      error = result.ErrorMessage();
      return result.Code();
    }

    result = pRuntime->Initialize();
    if (!result.IsOK()) {
      error = result.ErrorMessage();
      return result.Code();
    }

    return 0;
  }

  void FillInBatchSize(std::vector<int64_t>& shape, int inputSize, int featSize) {
    if ((inputSize % featSize != 0) && (featSize != -1))
      throw std::runtime_error("Input count is not a multiple of dimension.");

    int batchSize = featSize == -1 ? 1 : inputSize / featSize;
    shape.insert(shape.begin(), batchSize);
  }

  int ReadTensorStrings(Lotus::AllocatorPtr alloc, Lotus::MLValue* p_mlvalue,
                        TestDataReader& inputsReader, int featSize, std::vector<int64_t> dims, bool variableBatchSize) {
    using namespace Lotus;

    auto vec = inputsReader.GetSampleStrings(featSize, variableBatchSize);

    std::vector<std::string> vec2;
    for (int i = 0; i < vec.size(); i++) {
      std::string str(vec[i].begin(), vec[i].end());
      vec2.push_back(str);
    }

    if (variableBatchSize)
      FillInBatchSize(dims, gsl::narrow_cast<int>(vec.size()), featSize);

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

    p_mlvalue->Init(p_tensor.release(),
                    DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    return 0;
  }

  template <typename T>
  int ReadTensor(Lotus::AllocatorPtr alloc, Lotus::MLValue* p_mlvalue,
                 TestDataReader& inputsReader, int featSize, std::vector<int64_t> dims, bool variableBatchSize) {
    using namespace Lotus;

    auto vec = inputsReader.GetSample<T>(featSize, variableBatchSize);
    if (variableBatchSize)
      FillInBatchSize(dims, gsl::narrow_cast<int>(vec.size()), featSize);

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

    p_mlvalue->Init(p_tensor.release(),
                    Lotus::DataTypeImpl::GetType<Lotus::Tensor>(),
                    Lotus::DataTypeImpl::GetType<Lotus::Tensor>()->GetDeleteFunc());

    return 0;
  }

  template <typename V>
  int ReadTensorForMap(Lotus::AllocatorPtr alloc,
                       Lotus::MLValue* /*p_mlvalue*/, TestDataReader& inputsReader) {
    auto vec = inputsReader.GetSample<V>(-1);

    /*
        Lotus::Tensor *tensor = new Lotus::Tensor();
        tensor->Resize(1);
        auto* data = tensor->mutable_data<std::map<std::string, V>>();
        for (int i = 0; i < vec.size(); i++)
        {
            std::string s = std::to_string(i + 1);
            (*data)[s] = vec[i];
        }

        */
    return 0;
  }

  int Run(TestDataReader& inputsReader) {
    using namespace Lotus;
    int hr = 0;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // Create CPU input tensors
    Lotus::NameMLValMap feed;
    inputsReader.BufferNextSample();
    if (inputsReader.Eof())
      return 0;

    bool variableBatchSize = false;
    auto inputs_pairs = pRuntime->GetInputs();
    if (!inputs_pairs.first.IsOK()) {
      auto error = inputs_pairs.first.ErrorMessage();
      return inputs_pairs.first.Code();
    }

    auto& inputs = *(inputs_pairs.second);
    for (size_t index = 0; index < inputs.size(); index++) {
      MLValue mlvalue;
      const LotusIR::NodeArg& input = *(inputs[index]);
      const TensorShapeProto* input_shape = input.Shape();
      if (input.Name().empty())
        continue;

      auto type = input.Type();

      std::vector<int64_t> shape;
      int featSize = -1;

      //Previous graph input was variable length that consumed entire input line so fetch new input line.
      if (variableBatchSize)
        inputsReader.BufferNextSample();

      //This graph input may or may not be variable length.
      //REVIEW mzs: this can cause issues if we had variable-input followed by fixed input followed by variable-input where
      //fixed-input consumed all of the input line. *Ideally each graph input should be on its own line*.
      variableBatchSize = false;

      //If the shape is not available then read everything into the input tensor.
      //featSize = -1 indicates this condition.
      if (input_shape) {
        featSize = 0;
        auto dims = input_shape->dim();
        for (auto dim : dims) {
          if (dim.has_dim_param())
            variableBatchSize = true;
          else {
            auto dim_value = dim.dim_value();
            shape.push_back(dim_value);
            featSize = gsl::narrow_cast<int>(featSize ? featSize * dim_value : dim_value);
          }
        }
      }

      //REVIEW mzs: Here an assumption is made that all the input columns are for the map.
      //The supported map types in Lotus seen so far are <string, string> or <string, int64>.
      if (*type == "map(string,tensor(int64))") {
        ReadTensorForMap<int64_t>(TestCPUExecutionProvider()->GetAllocator(), &mlvalue, inputsReader);
        feed.insert(std::make_pair(input.Name(), mlvalue));
      } else if (*type == "map(string,tensor(float))" || *type == "map(string,tensor(double))") {
        ReadTensorForMap<float>(TestCPUExecutionProvider()->GetAllocator(), &mlvalue, inputsReader);
        feed.insert({input.Name(), mlvalue});
      } else {
        if (*type == "tensor(double)" || *type == "tensor(float)") {
          // If double is used in the following statement, following error occurs.
          // Tensor type mismatch, caller expects elements to be float while tensor contains double Error from operator
          ReadTensor<float>(TestCPUExecutionProvider()->GetAllocator(), &mlvalue, inputsReader, featSize, shape, variableBatchSize);
        } else if (*type == "tensor(int64)")
          ReadTensor<int64_t>(TestCPUExecutionProvider()->GetAllocator(), &mlvalue, inputsReader, featSize, shape, variableBatchSize);
        else if (*type == "tensor(string)")
          ReadTensorStrings(TestCPUExecutionProvider()->GetAllocator(), &mlvalue, inputsReader, featSize, shape, variableBatchSize);
        else
          throw std::runtime_error("Unsupported input type in LotusRT_exec" + std::string(*type));

        feed.insert(std::make_pair(input.Name(), mlvalue));
      }
    }

    // Create output feed
    std::vector<std::string> output_names;
    for (auto const& outp : *(pRuntime->GetOutputs().second)) {
      output_names.push_back(outp->Name());
    }

    std::string separator = "";
    // Invoke the net
    std::vector<Lotus::MLValue> outputMLValue;
    Lotus::Common::Status result = pRuntime->Run(feed, &outputMLValue);
    if (result.IsOK()) {
      auto outputMeta = pRuntime->GetOutputs().second;
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

 protected:
  Lotus::InferenceSession* pRuntime;

  static Lotus::IExecutionProvider* TestCPUExecutionProvider() {
    static Lotus::CPUExecutionProviderInfo info;
    static Lotus::CPUExecutionProvider cpu_provider(info);
    return &cpu_provider;
  }
};
