#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <random>
#include <vector>
#include "ternix/mul_mat_ternary.h"

// Generate random int8 activations
std::vector<int8_t> generate_activations(size_t size) {
    std::vector<int8_t> data(size);
    std::mt19937 engine(42);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<int8_t>(dist(engine));
    }
    return data;
}

static void BM_NaiveTernary(benchmark::State& state) {
    const size_t M = 4096;
    const size_t K = 4096;
    const size_t N = 4096;
    
    int8_t* A = static_cast<int8_t*>(_mm_malloc(M * K * sizeof(int8_t), 32));
    uint8_t* B = static_cast<uint8_t*>(_mm_malloc((K / 4) * N * sizeof(uint8_t), 32));
    int32_t* C = static_cast<int32_t*>(_mm_malloc(M * N * sizeof(int32_t), 32));
    
    for (auto _ : state) {
        ternix::mul_mat_ternary_naive(A, B, C, M, K, N);
        benchmark::DoNotOptimize(C);
    }
    
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
}
BENCHMARK(BM_NaiveTernary)->Unit(benchmark::kMillisecond);

static void BM_AVX2Ternary(benchmark::State& state) {
    const size_t M = 4096;
    const size_t K = 4096;
    const size_t N = 4096;
    
    int8_t* A = static_cast<int8_t*>(_mm_malloc(M * K * sizeof(int8_t), 32));
    uint8_t* B = static_cast<uint8_t*>(_mm_malloc((K / 4) * N * sizeof(uint8_t), 32));
    int32_t* C = static_cast<int32_t*>(_mm_malloc(M * N * sizeof(int32_t), 32));
    
    for (auto _ : state) {
        ternix::mul_mat_ternary_avx2(A, B, C, M, K, N);
        benchmark::DoNotOptimize(C);
    }
    
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
}
BENCHMARK(BM_AVX2Ternary)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
