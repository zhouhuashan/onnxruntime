#include <benchmark/benchmark.h>
#include <core/common/logging/logging.h>
#include <core/platform/env.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#include <core/framework/environment.h>
#include <core/common/logging/sinks/clog_sink.h>
using namespace Lotus;

static void BM_CPUAllocator(benchmark::State& state) {
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  const size_t len = state.range(0);
  for (auto _ : state) {
    void* p = cpu_allocator->Alloc(len);
    cpu_allocator->Free(p);
  }
}
BENCHMARK(BM_CPUAllocator)->Arg(4)->Arg(sizeof(Tensor));

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  std::string default_logger_id{"Default"};
  Logging::LoggingManager default_logging_manager{std::unique_ptr<Logging::ISink>{new Logging::CLogSink{}},
                                                  Logging::Severity::kWARNING, false,
                                                  Logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  std::unique_ptr<Environment> env;
  auto status = Environment::Create(env);
  ::benchmark::RunSpecifiedBenchmarks();
}
