cmake_minimum_required(VERSION 3.20)
project(TernixEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

find_package(OpenMP REQUIRED)

include(FetchContent)

# Google Benchmark
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

add_library(ternix_core STATIC
    src/mul_mat_ternary.cpp
)
target_include_directories(ternix_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ternix_core PUBLIC OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ternix_core PRIVATE -mavx2 -O3)
elseif(MSVC)
    target_compile_options(ternix_core PRIVATE /arch:AVX2 /O2)
endif()

# Benchmarks
add_executable(bench_simd_math benchmarks/bench_simd_math.cpp)
target_link_libraries(bench_simd_math PRIVATE ternix_core benchmark::benchmark benchmark::benchmark_main)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_simd_math PRIVATE -mavx2 -O3)
elseif(MSVC)
    target_compile_options(bench_simd_math PRIVATE /arch:AVX2 /O2)
endif()
