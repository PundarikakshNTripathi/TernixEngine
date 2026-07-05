#pragma once
#include <cstddef>
#include <cstdint>

namespace ternix {

/**
 * @brief Naive, unoptimized implementation of ternary matrix multiplication
 * 
 * Computes C = A * B, where:
 * @param A        M x K matrix of INT8 activations
 * @param B_packed K x N matrix of 1.58-bit (packed to 2-bit) weights
 * @param C        M x N matrix of INT32 accumulated results
 * @param M        Number of rows in A and C
 * @param K        Number of columns in A, rows in B
 * @param N        Number of columns in B and C
 */
void mul_mat_ternary_naive(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N
);

/**
 * @brief AVX2 SIMD-native implementation of ternary matrix multiplication
 * 
 * Bypasses floating point math. Uses in-register unpacking and accumulation.
 */
void mul_mat_ternary_avx2(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N
);

} // namespace ternix
