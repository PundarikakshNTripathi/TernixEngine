#pragma once
#include <cstdint>

namespace ternix {
    // Host wrapper for CUDA kernel
    void run_mul_mat_ternary_cuda(const float* A, const uint8_t* B_packed, float* C, int M, int K, int N);
}
