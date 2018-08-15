#include <benchmark/benchmark.h>
#include <core/graph/model.h>

static void BM_LoadModel(benchmark::State& state) {
  for (auto _ : state) {
    std::shared_ptr<LotusIR::Model> yolomodel;
    auto st = LotusIR::Model::Load("../models/test_tiny_yolov2/model.onnx", yolomodel);
    if (!st.IsOK()) {
      state.SkipWithError(st.ErrorMessage().c_str());
      break;
    }
  }
}

BENCHMARK(BM_LoadModel);