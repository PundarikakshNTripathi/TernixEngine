#include <benchmark/benchmark.h>
#include "ternix/model.hpp"

static void BM_EndToEndInference(benchmark::State& state) {
    // Stub for full layer inference benchmark
    for (auto _ : state) {
        // Simulate forward pass
    }
}
BENCHMARK(BM_EndToEndInference)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
