/*
* Copyright (c) Microsoft Corporation. All rights reserved.
*/

#pragma once
#include <iostream>
#include <fstream>
#include <vector>

inline std::string wstr2str(const std::wstring& wstr) {
  std::string str = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(wstr);
  return str;
}

class TestDataReader {
  std::wstring line;
  std::wifstream readerStream;
  std::wstringstream* rowStream = nullptr;

 public:
  TestDataReader(){};
  virtual ~TestDataReader() {
    if (rowStream) {
      delete (rowStream);
      rowStream = nullptr;
    }
  };

  void BufferNextSample();
  bool Eof();
  template <typename T>
  std::vector<T> GetSample(int smapleCount, bool variableBatchSize = false);
  std::vector<std::wstring> GetSampleStrings(int sampleCount, bool variableBatchSize = false);
  static TestDataReader* OpenReader(std::wstring dataFile);
};

bool TestDataReader::Eof() {
  return readerStream.eof();
}

void TestDataReader::BufferNextSample() {
  if (rowStream) {
    delete (rowStream);
    rowStream = nullptr;
  }

  if (Eof())
    return;

  std::getline(readerStream, line);

  if (Eof())
    return;

  rowStream = new std::wstringstream(line);
  std::wstring feature;
  std::getline(*rowStream, feature, L',');  //Skip the Label which is actually.
}

template <typename T>
std::vector<T> TestDataReader::GetSample(int sampleCount, bool variableBatchSize) {
  assert(sampleCount == -1 || sampleCount > 0);

  std::wstring feature;
  std::vector<T> result;

  int s = 0;
  while ((s++ < sampleCount || sampleCount == -1 || variableBatchSize) && std::getline(*rowStream, feature, L','))  // -1 means read all data in the sample
  {
    T featureValue;
    std::wstringstream featureConvert(feature);
    featureConvert >> featureValue;
    if (featureConvert.fail()) {
      featureValue = (T)NAN;
    }

    result.push_back(featureValue);
  }

  if (line.length() > 0 && line.back() == L',')
    result.push_back((T)NAN);

  if (sampleCount != -1 && !variableBatchSize) {
    //Remove the last NAN inserted if it is not part of this feature.
    if (result.size() == sampleCount + 1)
      result.pop_back();

    if (result.size() != sampleCount)
      throw std::runtime_error("Not enough features in sample.");
  }

  if (variableBatchSize && (result.size() % sampleCount != 0) && (sampleCount != -1))
    throw std::runtime_error("Input count is not a multiple of dimension.");

  return result;
}

std::vector<std::wstring> TestDataReader::GetSampleStrings(int sampleCount, bool variableBatchSize) {
  std::wstring feature;
  std::vector<std::wstring> result;

  int s = 0;
  while (s < sampleCount || sampleCount == -1 || variableBatchSize)  // -1 means read all data in the sample
  {
    if (std::getline(*rowStream, feature, L','))
      result.push_back(feature);
    else {
      if (sampleCount == -1 || variableBatchSize)
        break;

      throw std::runtime_error("Not enough features in sample.");
    }
    s++;
  }

  if (line.length() > 0 && line.back() == L',')
    result.push_back(L"");

  if (variableBatchSize && (result.size() % sampleCount != 0) && (sampleCount != -1))
    throw std::runtime_error("Input count is not a multiple of dimension.");

  return result;
}

TestDataReader* TestDataReader::OpenReader(std::wstring dataFile) {
  auto* reader = new TestDataReader();

  reader->readerStream.open(wstr2str(dataFile));
  if (!reader->readerStream) {
    delete (reader);
    return nullptr;
  }

  return reader;
}
